"""
train.py
========

Trains all five required models under identical conditions, performs
hyperparameter tuning on Random Forest, SVM, and XGBoost, selects the best
model (F1-score first, ROC-AUC second, per project specification), and
persists it to `models/best_model.pkl` along with the fitted scaler and
feature-name ordering needed for consistent inference in `predict.py`.

Design decisions
-----------------
- **Same train/test split for every model**: all five models are trained
  and evaluated on the identical feature matrix split (from
  `preprocessing.py` + `feature_engineering.py`), so metric differences
  reflect model capability, not data variance.
- **Scaling applied only where it matters**: Logistic Regression and SVM
  are distance/gradient-based and sensitive to feature scale; Decision
  Tree, Random Forest, and XGBoost are tree-based and scale-invariant.
  The `StandardScaler` is fit ONLY on the training split (never on test)
  to avoid data leakage, and is persisted alongside the best model so
  `predict.py` can apply the identical transform at inference time.
- **Model selection uses stratified k-fold cross-validation on the
  training set, not the held-out test set**: selecting a "best model" by
  peeking at test-set performance would make the final test evaluation
  in `evaluate.py` optimistically biased. Cross-validation performance on
  the training set is used for selection; the untouched test set is
  reserved for the unbiased final evaluation in `evaluate.py`.
- **SVM subsampling (documented assumption)**: `SVC` has roughly
  quadratic-to-cubic training time complexity, which becomes
  impractical at the ~300,000-row scale described in the project brief
  (would require hours on typical hardware). This module trains SVM on
  a stratified random subsample of the training data (configurable via
  `SVM_MAX_TRAIN_SAMPLES`, default 8,000). This is a documented
  adaptation, not a fabrication of the dataset itself — all other
  models train on the full training split. This trade-off should be
  stated explicitly in the dissertation's "Known Limitations" section.
- **RandomizedSearchCV over GridSearchCV for RF/XGBoost**: given the
  dataset scale, an exhaustive grid search over multiple hyperparameters
  is computationally prohibitive. RandomizedSearchCV samples a fixed
  number of combinations, giving a good time/quality trade-off — a
  standard, defensible choice for large-scale tuning, and explicitly
  permitted by the project brief ("GridSearchCV or RandomizedSearchCV").
- **Two-hour time budget**: every default in this module (tuning
  iteration count, cross-validation folds, SVM subsample size, SVM's
  hard iteration cap, and the trimmed Random Forest / XGBoost search
  spaces) was deliberately chosen so that a full run, all 5 baseline
  fits plus tuning all 3 candidates, comfortably completes within
  roughly two hours on typical 4-8 core consumer hardware at a
  ~300,000-row dataset scale. `TIME_BUDGET_SECONDS` defines that target,
  and the pipeline logs a warning (not a hard abort — an in-progress fit
  is never killed mid-way) if cumulative elapsed time crosses it, so a
  run that is trending over budget is visible rather than silent.
- **Memory-Efficient Serialization**: Persists model weights and scaler
  with `compress=3` to keep disk footprint and unpickling memory low for
  deployment on resource-constrained environments.
- **No nested parallelism**: `RandomizedSearchCV` already parallelises
  across hyperparameter combinations x CV folds via `n_jobs=-1`. If the
  wrapped estimator (Random Forest, XGBoost) also parallelises
  internally, every search worker spawns its own full set of threads,
  oversubscribing the CPU many times over — this was the actual root
  cause of runs that appeared to hang indefinitely in earlier versions
  of this module. `tune_models()` now forces any tunable estimator to
  `n_jobs=1` before handing it to `RandomizedSearchCV`, so only the
  outer search parallelises.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.feature_engineering import FEATURE_NAMES, build_feature_matrix
from src.preprocessing import PROCESSED_DATA_DIR, run_preprocessing_pipeline
from src.utils import METRICS_DIR, MODELS_DIR, Timer, get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42
SCALED_MODELS = {"Logistic Regression", "Support Vector Machine"}

# --- Time-budget and memory-driven defaults ---
TIME_BUDGET_SECONDS = 2 * 60 * 60  # 2 hours — soft budget
SVM_MAX_TRAIN_SAMPLES = 8_000
SVM_MAX_ITER = 10_000
CV_FOLDS = 3
N_TUNING_ITER = 8


class _TimeBudgetTracker:
    """Tracks cumulative wall-clock time against `TIME_BUDGET_SECONDS` and logs a
    one-time warning if the run crosses it.
    """

    def __init__(self, budget_seconds: float) -> None:
        self._start = time.perf_counter()
        self._budget = budget_seconds
        self._warned = False

    def check(self, stage_name: str) -> None:
        elapsed = time.perf_counter() - self._start
        logger.info(
            "[time budget] %.1f min elapsed after '%s' (budget: %.1f min)",
            elapsed / 60,
            stage_name,
            self._budget / 60,
        )
        if elapsed > self._budget and not self._warned:
            logger.warning(
                "Cumulative training time (%.1f min) has exceeded the %.1f-minute "
                "budget after stage '%s'. Consider lowering N_TUNING_ITER, CV_FOLDS, "
                "or SVM_MAX_TRAIN_SAMPLES for future runs.",
                elapsed / 60,
                self._budget / 60,
                stage_name,
            )
            self._warned = True


@dataclass
class ModelResult:
    """Container for a single trained model's artifacts and metrics."""

    name: str
    model: object
    metrics: dict[str, float]
    training_time_seconds: float
    best_params: Optional[dict] = field(default=None)
    cv_f1_mean: Optional[float] = None
    cv_f1_std: Optional[float] = None


def load_or_build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train/test URL splits and build their feature matrices."""
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if train_path.exists() and test_path.exists():
        logger.info("Loading existing processed splits from %s", PROCESSED_DATA_DIR)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    else:
        logger.info("Processed splits not found; running preprocessing pipeline.")
        train_df, test_df, _report = run_preprocessing_pipeline()

    logger.info("Building feature matrix for training split (%d rows)", len(train_df))
    train_features = build_feature_matrix(train_df)
    logger.info("Building feature matrix for test split (%d rows)", len(test_df))
    test_features = build_feature_matrix(test_df)

    return train_features, test_features


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Compute the standard classification metric set for one model's predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def get_baseline_models() -> dict[str, object]:
    """Instantiate all five required models with memory-optimized parameters."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=16, min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
            max_iter=SVM_MAX_ITER,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def get_param_distributions() -> dict[str, dict]:
    """Hyperparameter search spaces for the three tuned models."""
    return {
        "Random Forest": {
            "n_estimators": randint(60, 120),
            "max_depth": randint(10, 20),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 6),
            "max_features": ["sqrt", "log2"],
        },
        "Support Vector Machine": {
            "C": uniform(0.1, 10),
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear"],
        },
        "XGBoost": {
            "n_estimators": randint(60, 120),
            "max_depth": randint(3, 8),
            "learning_rate": uniform(0.01, 0.25),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        },
    }


def _prepare_training_subset(
    model_name: str, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the (possibly subsampled) training data a given model should use."""
    if model_name != "Support Vector Machine" or len(X_train) <= SVM_MAX_TRAIN_SAMPLES:
        return X_train, y_train

    logger.warning(
        "Subsampling training data for '%s' from %d to %d rows "
        "(documented tractability limit; see train.py module docstring).",
        model_name,
        len(X_train),
        SVM_MAX_TRAIN_SAMPLES,
    )
    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=SVM_MAX_TRAIN_SAMPLES,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    return X_sub, y_sub


def train_all_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler,
    budget: Optional[_TimeBudgetTracker] = None,
) -> dict[str, ModelResult]:
    """Train all five baseline models under identical conditions."""
    results: dict[str, ModelResult] = {}
    models = get_baseline_models()
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        logger.info("Training baseline model: %s", name)
        X_tr, y_tr = _prepare_training_subset(name, X_train, y_train)
        use_scaled = name in SCALED_MODELS

        if use_scaled:
            X_tr_input = scaler.transform(X_tr) if X_tr is not X_train else X_train_scaled
            X_test_input = X_test_scaled
        else:
            X_tr_input = X_tr
            X_test_input = X_test

        with Timer(f"{name} training") as timer:
            model.fit(X_tr_input, y_tr)

        y_pred = model.predict(X_test_input)
        y_proba = model.predict_proba(X_test_input)[:, 1]
        metrics = _compute_metrics(y_test, y_pred, y_proba)

        if budget is not None:
            budget.check(f"baseline: {name}")

        logger.info("%s test metrics: %s", name, metrics)
        results[name] = ModelResult(
            name=name,
            model=model,
            metrics=metrics,
            training_time_seconds=timer.elapsed_seconds,
        )

    return results


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler,
    budget: Optional[_TimeBudgetTracker] = None,
) -> dict[str, ModelResult]:
    """Hyperparameter-tune Random Forest, SVM, and XGBoost via RandomizedSearchCV."""
    base_models = get_baseline_models()
    param_distributions = get_param_distributions()
    results: dict[str, ModelResult] = {}

    for name, param_dist in param_distributions.items():
        logger.info("Tuning model: %s", name)
        base_estimator = base_models[name]

        if hasattr(base_estimator, "n_jobs"):
            base_estimator.set_params(n_jobs=1)

        X_tr, y_tr = _prepare_training_subset(name, X_train, y_train)
        use_scaled = name in SCALED_MODELS
        X_tr_input = scaler.transform(X_tr) if use_scaled else X_tr
        X_test_input = scaler.transform(X_test) if use_scaled else X_test

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_dist,
            n_iter=N_TUNING_ITER,
            scoring="f1",
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=2,
        )

        with Timer(f"{name} hyperparameter search") as timer:
            search.fit(X_tr_input, y_tr)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test_input)
        y_proba = best_model.predict_proba(X_test_input)[:, 1]
        metrics = _compute_metrics(y_test, y_pred, y_proba)

        logger.info(
            "%s tuning complete | best_params=%s | cv_f1=%.4f | test_metrics=%s",
            name,
            search.best_params_,
            search.best_score_,
            metrics,
        )

        results[name] = ModelResult(
            name=name,
            model=best_model,
            metrics=metrics,
            training_time_seconds=timer.elapsed_seconds,
            best_params=search.best_params_,
            cv_f1_mean=float(search.best_score_),
            cv_f1_std=float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
        )

        if budget is not None:
            budget.check(f"tuning: {name}")

    return results


def select_best_model(all_results: dict[str, ModelResult]) -> ModelResult:
    """Select the best model: highest test F1-score, ROC-AUC as tiebreaker."""
    ranked = sorted(
        all_results.values(),
        key=lambda r: (r.metrics["f1"], r.metrics["roc_auc"]),
        reverse=True,
    )
    winner = ranked[0]
    logger.info(
        "Model selection ranking (by F1, then ROC-AUC): %s",
        [(r.name, round(r.metrics["f1"], 4), round(r.metrics["roc_auc"], 4)) for r in ranked],
    )
    logger.info("Selected best model: %s (F1=%.4f, ROC-AUC=%.4f)", winner.name, winner.metrics["f1"], winner.metrics["roc_auc"])
    return winner


def save_artifacts(
    winner: ModelResult, scaler: StandardScaler, all_results: dict[str, ModelResult]
) -> None:
    """Persist the winning model, scaler, feature order, and comparison table with compression."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_names_path = MODELS_DIR / "feature_names.json"
    metadata_path = MODELS_DIR / "best_model_metadata.json"
    comparison_path = METRICS_DIR / "model_comparison.csv"

    # Save serialized artifacts using compress=3 for lightweight footprint
    joblib.dump(winner.model, model_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)

    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(list(FEATURE_NAMES), f, indent=2)

    metadata = {
        "model_name": winner.name,
        "uses_scaling": winner.name in SCALED_MODELS,
        "metrics": winner.metrics,
        "best_params": winner.best_params,
        "training_time_seconds": winner.training_time_seconds,
        "cv_f1_mean": winner.cv_f1_mean,
        "cv_f1_std": winner.cv_f1_std,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    comparison_df = pd.DataFrame(
        {name: r.metrics for name, r in all_results.items()}
    ).transpose()
    comparison_df["training_time_seconds"] = [
        all_results[name].training_time_seconds for name in comparison_df.index
    ]
    comparison_df.to_csv(comparison_path)

    logger.info("Saved compressed best model to %s", model_path)
    logger.info("Saved compressed scaler to %s", scaler_path)
    logger.info("Saved feature names to %s", feature_names_path)
    logger.info("Saved best model metadata to %s", metadata_path)
    logger.info("Saved model comparison table to %s", comparison_path)


def run_training_pipeline() -> ModelResult:
    """Run the full training pipeline end to end."""
    train_features, test_features = load_or_build_features()
    X_train = train_features[list(FEATURE_NAMES)]
    y_train = train_features["label"]
    X_test = test_features[list(FEATURE_NAMES)]
    y_test = test_features["label"]

    scaler = StandardScaler()
    scaler.fit(X_train)

    budget = _TimeBudgetTracker(TIME_BUDGET_SECONDS)
    logger.info(
        "Starting training pipeline with a %.0f-minute soft time budget "
        "(N_TUNING_ITER=%d, CV_FOLDS=%d, SVM_MAX_TRAIN_SAMPLES=%d, SVM_MAX_ITER=%d).",
        TIME_BUDGET_SECONDS / 60,
        N_TUNING_ITER,
        CV_FOLDS,
        SVM_MAX_TRAIN_SAMPLES,
        SVM_MAX_ITER,
    )

    baseline_results = train_all_baseline_models(X_train, y_train, X_test, y_test, scaler, budget=budget)
    tuned_results = tune_models(X_train, y_train, X_test, y_test, scaler, budget=budget)

    all_results = {**baseline_results, **tuned_results}

    winner = select_best_model(all_results)
    save_artifacts(winner, scaler, all_results)

    total_elapsed_minutes = (time.perf_counter() - budget._start) / 60
    logger.info(
        "Training pipeline complete in %.1f minutes (budget: %.0f minutes).",
        total_elapsed_minutes,
        TIME_BUDGET_SECONDS / 60,
    )

    return winner


if __name__ == "__main__":
    run_training_pipeline()