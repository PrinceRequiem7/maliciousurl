"""
evaluate.py
===========

Performs the final, rigorous evaluation of the persisted best model
(`models/best_model.pkl`) on the held-out test set that `train.py`
deliberately never used for model selection (see `train.py` module
docstring). This separation matters: any metric computed here is an
unbiased estimate of real-world performance, whereas metrics used during
training/tuning/selection are optimistic by construction.

Outputs (all saved under results/):
  - results/metrics/test_set_metrics.json   (accuracy, precision, recall, F1, ROC-AUC)
  - results/metrics/classification_report.txt
  - results/metrics/cross_validation_scores.json
  - results/figures/confusion_matrix_<model>.png
  - results/figures/roc_curves_comparison.png   (single-model ROC, reused chart fn)
  - results/figures/precision_recall_curves_comparison.png
"""

from __future__ import annotations

import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import FEATURE_NAMES, build_feature_matrix
from src.train import CV_FOLDS, RANDOM_STATE, SCALED_MODELS
from src.utils import MODELS_DIR, PROCESSED_DATA_DIR, METRICS_DIR, get_logger
from src.visualization import (
    plot_confusion_matrix_chart,
    plot_precision_recall_curves,
    plot_roc_curves,
)

logger = get_logger(__name__)


def load_model_artifacts() -> tuple[object, StandardScaler, list[str], dict]:
    """Load the persisted best model, scaler, feature names, and metadata.

    Returns:
        Tuple of (model, scaler, feature_names, metadata_dict).

    Raises:
        FileNotFoundError: If `train.py` has not been run yet and no
            saved model exists.
    """
    model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_names_path = MODELS_DIR / "feature_names.json"
    metadata_path = MODELS_DIR / "best_model_metadata.json"

    for path in (model_path, scaler_path, feature_names_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Expected model artifact not found at '{path}'. "
                "Run `python -m src.train` first to train and save a model."
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("Loaded model '%s' and associated artifacts", metadata.get("model_name"))
    return model, scaler, feature_names, metadata


def load_test_features(feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load the test split and build its feature matrix.

    Args:
        feature_names: Ordered feature names to select as X columns
            (must match the order used at training time).

    Returns:
        Tuple of (X_test, y_test).

    Raises:
        FileNotFoundError: If `data/processed/test.csv` does not exist.
    """
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Processed test set not found at '{test_path}'. "
            "Run preprocessing/training first."
        )
    test_df = pd.read_csv(test_path)
    test_features = build_feature_matrix(test_df)
    X_test = test_features[feature_names]
    y_test = test_features["label"]
    return X_test, y_test


def load_train_features(feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load the training split and build its feature matrix (for cross-validation only)."""
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Processed train set not found at '{train_path}'.")
    train_df = pd.read_csv(train_path)
    train_features = build_feature_matrix(train_df)
    X_train = train_features[feature_names]
    y_train = train_features["label"]
    return X_train, y_train


def evaluate_on_test_set(
    model: object,
    scaler: StandardScaler,
    metadata: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute the full metric suite and predictions on the held-out test set.

    Args:
        model: The fitted best model.
        scaler: Scaler fit during training (applied only if the model
            needs it, per `metadata['uses_scaling']`).
        metadata: Model metadata dict from `train.py` (must include
            'model_name' and 'uses_scaling').
        X_test: Test feature matrix (unscaled).
        y_test: Test labels.

    Returns:
        Dict with keys: metrics (dict), y_true, y_pred, y_proba (arrays),
        model_name.
    """
    uses_scaling = metadata.get("uses_scaling", False)
    X_input = scaler.transform(X_test) if uses_scaling else X_test

    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    logger.info("Final held-out test set metrics for %s: %s", metadata["model_name"], metrics)

    return {
        "metrics": metrics,
        "y_true": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
        "y_proba": np.asarray(y_proba),
        "model_name": metadata["model_name"],
    }


def run_cross_validation(
    model: object,
    metadata: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scaler: StandardScaler,
    folds: int = CV_FOLDS,
) -> dict:
    """Run stratified k-fold cross-validation of the best model on the training set.

    Note: this refits fresh clones of the model on each fold internally
    (via `cross_val_score`); it does not reuse the already-fitted `model`
    object's learned weights, which is the correct way to estimate
    generalization variance.

    Args:
        model: The fitted best model (its class/hyperparameters are
            reused, not its fitted weights).
        metadata: Model metadata dict (for uses_scaling / model_name).
        X_train: Training feature matrix (unscaled).
        y_train: Training labels.
        scaler: Scaler fit on the training data.
        folds: Number of cross-validation folds.

    Returns:
        Dict mapping metric name -> {"mean": float, "std": float}.
    """
    from sklearn.base import clone

    uses_scaling = metadata.get("uses_scaling", False)
    X_input = scaler.transform(X_train) if uses_scaling else X_train
    cv_estimator = clone(model)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = ("accuracy", "precision", "recall", "f1", "roc_auc")

    results: dict[str, dict[str, float]] = {}
    for metric in scoring:
        scores = cross_val_score(cv_estimator, X_input, y_train, cv=skf, scoring=metric, n_jobs=-1)
        results[metric] = {"mean": float(scores.mean()), "std": float(scores.std())}
        logger.info("CV %s: mean=%.4f std=%.4f", metric, scores.mean(), scores.std())

    return results


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
    """Save a full sklearn classification report as a text file.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name of the evaluated model (for the report header).

    Returns:
        Path to the saved report file.
    """
    report_text = classification_report(
        y_true, y_pred, target_names=["Safe", "Malicious"], zero_division=0
    )
    path = METRICS_DIR / "classification_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report — {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)
    logger.info("Saved classification report to %s", path)
    return str(path)


def run_evaluation_pipeline() -> dict:
    """Run the full evaluation pipeline end to end.

    1. Load best model artifacts.
    2. Evaluate on the held-out test set (metrics + predictions).
    3. Run stratified k-fold cross-validation on the training set.
    4. Save classification report, confusion matrix, ROC curve, PR curve.
    5. Save all metrics/CV results as JSON.

    Returns:
        Dict summarizing everything computed (metrics, cv_results, paths).
    """
    model, scaler, feature_names, metadata = load_model_artifacts()
    X_test, y_test = load_test_features(feature_names)
    X_train, y_train = load_train_features(feature_names)

    test_results = evaluate_on_test_set(model, scaler, metadata, X_test, y_test)
    cv_results = run_cross_validation(model, metadata, X_train, y_train, scaler)

    report_path = save_classification_report(
        test_results["y_true"], test_results["y_pred"], test_results["model_name"]
    )
    cm_path = plot_confusion_matrix_chart(
        test_results["y_true"], test_results["y_pred"], test_results["model_name"]
    )
    curve_data = {
        test_results["model_name"]: {
            "y_true": test_results["y_true"],
            "y_proba": test_results["y_proba"],
        }
    }
    roc_path = plot_roc_curves(curve_data)
    pr_path = plot_precision_recall_curves(curve_data)

    metrics_path = METRICS_DIR / "test_set_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model_name": test_results["model_name"], "metrics": test_results["metrics"]},
            f,
            indent=2,
        )

    cv_path = METRICS_DIR / "cross_validation_scores.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    logger.info("Evaluation pipeline complete.")
    return {
        "model_name": test_results["model_name"],
        "test_metrics": test_results["metrics"],
        "cv_results": cv_results,
        "paths": {
            "classification_report": report_path,
            "confusion_matrix": cm_path,
            "roc_curve": roc_path,
            "pr_curve": pr_path,
            "metrics_json": str(metrics_path),
            "cv_json": str(cv_path),
        },
    }


if __name__ == "__main__":
    summary = run_evaluation_pipeline()
    logger.info("Evaluation summary: %s", summary)
