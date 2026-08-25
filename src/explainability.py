"""
explainability.py
==================

Generates SHAP-based explanations for the persisted best model, answering
the project requirement: "Explain which URL characteristics contribute
most to malicious classification."

Outputs (saved under results/):
  - results/figures/shap_global_importance.png
  - results/figures/shap_summary_plot.png
  - results/figures/shap_waterfall_malicious_example.png
  - results/metrics/shap_top_features.json (textual ranking, for the report)

Design notes
------------
- **Explainer selection is model-type-aware**: tree-based models
  (Decision Tree, Random Forest, XGBoost) use `shap.TreeExplainer`,
  which computes EXACT Shapley values efficiently by exploiting tree
  structure. Non-tree models (Logistic Regression, SVM) fall back to
  the model-agnostic `shap.Explainer` wrapping `predict_proba`, which
  uses a permutation-based approximation. Both paths are handled so
  explainability works regardless of which model `train.py` selects as
  the winner.
- **Sampling for tractability**: explanations are computed on a random
  sample of the test set (default 200 rows) with a smaller background
  reference set (default 100 rows), rather than the full ~300,000-row
  test set. SHAP computation cost scales with sample size, and a few
  hundred rows is more than sufficient to produce stable global
  importance rankings and representative plots — a standard practice
  in applied SHAP analysis.
- **Malicious-class perspective**: for binary classifiers, both
  explainer paths return SHAP values for both classes. This module
  consistently extracts the values for class 1 (Malicious), since the
  project asks specifically which characteristics drive malicious
  classification.
"""

from __future__ import annotations

import json
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.evaluate import load_model_artifacts, load_test_features, load_train_features
from src.utils import FIGURES_DIR, METRICS_DIR, get_logger

logger = get_logger(__name__)

TREE_BASED_CLASS_NAMES = {"DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier"}
DEFAULT_BACKGROUND_SIZE = 100
DEFAULT_SAMPLE_SIZE = 200
RANDOM_STATE = 42


def build_explainer(
    model: object, scaler, uses_scaling: bool, X_background: pd.DataFrame
) -> tuple[object, str]:
    """Construct a SHAP explainer appropriate to the model's type.

    Args:
        model: The fitted best model.
        scaler: Fitted StandardScaler (used to transform the background
            set for models that require scaled input).
        uses_scaling: Whether this model was trained on scaled features.
        X_background: Background reference data (unscaled); used as the
            reference distribution for the model-agnostic explainer, and
            ignored (but accepted for a uniform call signature) by
            TreeExplainer.

    Returns:
        Tuple of (explainer, kind) where kind is "tree" or "generic".
    """
    model_class_name = type(model).__name__

    if model_class_name in TREE_BASED_CLASS_NAMES:
        logger.info("Using shap.TreeExplainer for model type '%s'", model_class_name)
        explainer = shap.TreeExplainer(model)
        return explainer, "tree"

    logger.info(
        "Using model-agnostic shap.Explainer (permutation-based) for model type '%s'",
        model_class_name,
    )
    background = scaler.transform(X_background) if uses_scaling else X_background.values
    explainer = shap.Explainer(model.predict_proba, background)
    return explainer, "generic"


def compute_shap_values(
    explainer: object,
    kind: str,
    X_sample: pd.DataFrame,
    scaler,
    uses_scaling: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for the Malicious (class 1) output.

    Args:
        explainer: A fitted explainer from `build_explainer`.
        kind: "tree" or "generic", as returned by `build_explainer`.
        X_sample: Rows to explain (unscaled).
        scaler: Fitted StandardScaler.
        uses_scaling: Whether the model requires scaled input.

    Returns:
        Tuple of (shap_values, base_values):
          - shap_values: ndarray of shape (n_samples, n_features), SHAP
            values for the Malicious class.
          - base_values: ndarray of shape (n_samples,), the explainer's
            expected/base value for the Malicious class, broadcast per
            sample for convenience in waterfall plotting.
    """
    X_input = scaler.transform(X_sample) if uses_scaling else X_sample.values

    if kind == "tree":
        raw = explainer.shap_values(X_input)
        if isinstance(raw, list):
            values = np.asarray(raw[1])
            expected = explainer.expected_value
            base = expected[1] if isinstance(expected, (list, np.ndarray)) else expected
        elif np.ndim(raw) == 3:
            values = raw[:, :, 1]
            expected = explainer.expected_value
            base = expected[1] if isinstance(expected, (list, np.ndarray)) else expected
        else:
            values = raw
            base = explainer.expected_value
        base_values = np.full(shape=(values.shape[0],), fill_value=base, dtype=float)
    else:
        explanation = explainer(X_input)
        if explanation.values.ndim == 3:
            values = explanation.values[:, :, 1]
            base_values = np.asarray(explanation.base_values)[:, 1]
        else:
            values = explanation.values
            base_values = np.asarray(explanation.base_values)

    return np.asarray(values, dtype=float), np.asarray(base_values, dtype=float)


def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
    filename: str = "shap_global_importance.png",
) -> tuple[str, list[tuple[str, float]]]:
    """Bar chart of mean |SHAP value| per feature (global importance).

    Args:
        shap_values: Array of shape (n_samples, n_features).
        feature_names: Feature names, aligned to shap_values columns.
        top_n: Number of top features to display.
        filename: Output filename under results/figures/.

    Returns:
        Tuple of (saved figure path, ranked list of (feature, mean_abs_shap)).
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs), key=lambda pair: pair[1], reverse=True)
    top = ranking[:top_n][::-1]  # reversed for horizontal bar chart (largest on top)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([name for name, _ in top], [val for _, val in top], color="#4A4E69")
    ax.set_xlabel("Mean |SHAP value| (impact on Malicious prediction)")
    ax.set_title(f"Global Feature Importance (SHAP) — Top {top_n}")
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP global importance chart to %s", path)

    return str(path), ranking


def plot_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    filename: str = "shap_summary_plot.png",
) -> str:
    """Save a SHAP summary (beeswarm) plot.

    Args:
        shap_values: Array of shape (n_samples, n_features).
        X_sample: The corresponding (unscaled, for readability) feature
            values shown alongside the SHAP values.
        filename: Output filename under results/figures/.

    Returns:
        Saved figure path.
    """
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    path = FIGURES_DIR / filename
    plt.gcf().savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    logger.info("Saved SHAP summary plot to %s", path)
    return str(path)


def plot_waterfall_for_malicious_example(
    shap_values: np.ndarray,
    base_values: np.ndarray,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    predicted_labels: np.ndarray,
    feature_names: list[str],
    filename: str = "shap_waterfall_malicious_example.png",
) -> Optional[str]:
    """Save a SHAP waterfall plot for one correctly-predicted Malicious example.

    Picks the first sample in `X_sample` that is both truly Malicious and
    predicted Malicious, so the explanation illustrates a genuine,
    correctly-classified malicious case rather than an ambiguous or
    misclassified one.

    Args:
        shap_values: Array of shape (n_samples, n_features).
        base_values: Array of shape (n_samples,).
        X_sample: Feature values corresponding to shap_values rows.
        y_sample: True labels corresponding to shap_values rows.
        predicted_labels: Model predictions corresponding to shap_values rows.
        feature_names: Feature names.
        filename: Output filename under results/figures/.

    Returns:
        Saved figure path, or None if no correctly-predicted Malicious
        example was found in the sample.
    """
    y_sample_arr = np.asarray(y_sample)
    candidates = np.where((y_sample_arr == 1) & (predicted_labels == 1))[0]
    if len(candidates) == 0:
        logger.warning(
            "No correctly-predicted Malicious example found in the SHAP sample; "
            "skipping waterfall plot."
        )
        return None

    idx = int(candidates[0])
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=base_values[idx],
        data=X_sample.iloc[idx].values,
        feature_names=feature_names,
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation, show=False)
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved SHAP waterfall plot (test row index %d) to %s", idx, path)
    return str(path)


def run_explainability_pipeline(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
) -> dict:
    """Run the full SHAP explainability pipeline end to end.

    1. Load the best model, scaler, and feature names.
    2. Sample background (from train) and explanation (from test) rows.
    3. Build a model-type-appropriate explainer and compute SHAP values.
    4. Save global importance, summary, and waterfall plots.
    5. Save a JSON ranking of top contributing features.

    Args:
        sample_size: Number of test rows to explain.
        background_size: Number of training rows used as the background
            reference distribution.

    Returns:
        Dict with the top-feature ranking and all saved figure paths.
    """
    model, scaler, feature_names, metadata = load_model_artifacts()
    uses_scaling = metadata.get("uses_scaling", False)

    X_train, _y_train_full = load_train_features(feature_names)
    X_test, y_test = load_test_features(feature_names)

    background_size = min(background_size, len(X_train))
    sample_size = min(sample_size, len(X_test))
    X_background = X_train.sample(background_size, random_state=RANDOM_STATE)
    sample_idx = X_test.sample(sample_size, random_state=RANDOM_STATE).index
    X_sample = X_test.loc[sample_idx].reset_index(drop=True)
    y_sample = y_test.loc[sample_idx].reset_index(drop=True)

    X_sample_input = scaler.transform(X_sample) if uses_scaling else X_sample.values
    predicted_labels = model.predict(X_sample_input)

    explainer, kind = build_explainer(model, scaler, uses_scaling, X_background)
    shap_values, base_values = compute_shap_values(
        explainer, kind, X_sample, scaler, uses_scaling
    )

    importance_path, ranking = plot_global_importance(shap_values, feature_names)
    summary_path = plot_summary(shap_values, X_sample)
    waterfall_path = plot_waterfall_for_malicious_example(
        shap_values, base_values, X_sample, y_sample, predicted_labels, feature_names
    )

    top_features_path = METRICS_DIR / "shap_top_features.json"
    ranking_serializable = [{"feature": name, "mean_abs_shap": val} for name, val in ranking]
    with open(top_features_path, "w", encoding="utf-8") as f:
        json.dump(ranking_serializable, f, indent=2)

    logger.info(
        "Top 5 features driving Malicious classification: %s",
        [item["feature"] for item in ranking_serializable[:5]],
    )

    return {
        "model_name": metadata["model_name"],
        "explainer_kind": kind,
        "top_features": ranking_serializable[:10],
        "paths": {
            "global_importance": importance_path,
            "summary_plot": summary_path,
            "waterfall_plot": waterfall_path,
            "top_features_json": str(top_features_path),
        },
    }


if __name__ == "__main__":
    summary = run_explainability_pipeline()
    logger.info("Explainability summary: %s", summary)
