"""
visualization.py
=================

Centralized plotting utilities for both the EDA phase and the model
evaluation phase. Every function here does ONE plot, saves it as a PNG to
`results/figures/`, and returns the saved path — so `train.py`,
`evaluate.py`, `explainability.py`, and the notebooks can all reuse the
exact same plotting logic instead of duplicating matplotlib boilerplate.

Uses only the `matplotlib` + `pandas` dependencies already listed in the
project's requirements.txt (no seaborn), per the "do not introduce
unnecessary frameworks" constraint. Pair plots are produced with
`pandas.plotting.scatter_matrix`, which is matplotlib-backed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: safe for servers / CI / notebooks alike

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from src.utils import FIGURES_DIR, METRICS_DIR, get_logger

logger = get_logger(__name__)

# Consistent, colorblind-friendlier palette used across all charts.
_SAFE_COLOR = "#2E86AB"
_MALICIOUS_COLOR = "#C1121F"
_NEUTRAL_COLOR = "#4A4E69"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }
)


def _save_fig(fig: plt.Figure, filename: str) -> str:
    """Save a matplotlib figure to results/figures/ and close it.

    Args:
        fig: The matplotlib Figure to save.
        filename: Filename (with extension) to save under.

    Returns:
        The absolute path the figure was saved to, as a string.
    """
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return str(path)


# ------------------------------------------------------------------------
# EDA plots
# ------------------------------------------------------------------------
def plot_class_distribution(
    df: pd.DataFrame, label_col: str = "label", filename: str = "class_distribution.png"
) -> str:
    """Bar chart of Safe vs Malicious class counts."""
    counts = df[label_col].value_counts().sort_index()
    labels = ["Safe" if idx == 0 else "Malicious" for idx in counts.index]
    colors = [_SAFE_COLOR if idx == 0 else _MALICIOUS_COLOR for idx in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of URLs")
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
        )
    return _save_fig(fig, filename)


def plot_url_length_histogram(
    df: pd.DataFrame,
    length_col: str = "url_length",
    label_col: str = "label",
    filename: str = "url_length_histogram.png",
) -> str:
    """Overlaid histogram of URL length, split by Safe vs Malicious."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label_value, name, color in ((0, "Safe", _SAFE_COLOR), (1, "Malicious", _MALICIOUS_COLOR)):
        subset = df.loc[df[label_col] == label_value, length_col]
        ax.hist(subset, bins=50, alpha=0.55, label=name, color=color)
    ax.set_title("URL Length Distribution by Class")
    ax.set_xlabel("URL Length (characters)")
    ax.set_ylabel("Frequency")
    ax.legend()
    return _save_fig(fig, filename)


def plot_correlation_heatmap(
    feature_df: pd.DataFrame, filename: str = "correlation_heatmap.png"
) -> str:
    """Correlation heatmap across all numeric features."""
    numeric_df = feature_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save_fig(fig, filename)


def plot_missing_values(df: pd.DataFrame, filename: str = "missing_values.png") -> str:
    """Bar chart of missing-value counts per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.bar(missing.index, missing.values, color=_NEUTRAL_COLOR)
        ax.set_ylabel("Missing Count")
        plt.xticks(rotation=45, ha="right")
    ax.set_title("Missing Values by Column")
    return _save_fig(fig, filename)


def plot_feature_importance_preview(
    feature_df: pd.DataFrame,
    label_col: str = "label",
    top_n: int = 15,
    filename: str = "feature_importance_preview.png",
) -> str:
    """Preview of feature relevance via absolute Pearson correlation with the label.

    This is an EDA-stage *preview* only (cheap, model-free signal), distinct
    from the rigorous SHAP-based importance computed later in
    `explainability.py` once a model has been trained.
    """
    numeric_df = feature_df.select_dtypes(include=[np.number])
    if label_col not in numeric_df.columns:
        raise ValueError(f"'{label_col}' must be a numeric column in feature_df.")

    correlations = (
        numeric_df.corr()[label_col].drop(label_col).abs().sort_values(ascending=False)
    )
    top = correlations.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=_NEUTRAL_COLOR)
    ax.set_title(f"Top {top_n} Features by |Correlation| with Label (EDA Preview)")
    ax.set_xlabel("Absolute Pearson Correlation")
    return _save_fig(fig, filename)


def plot_boxplots(
    feature_df: pd.DataFrame,
    features: Sequence[str],
    label_col: str = "label",
    filename: str = "boxplots.png",
) -> str:
    """Grid of boxplots for the given features, split by class."""
    n = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        safe_vals = feature_df.loc[feature_df[label_col] == 0, feature]
        mal_vals = feature_df.loc[feature_df[label_col] == 1, feature]
        ax.boxplot(
            [safe_vals, mal_vals],
            tick_labels=["Safe", "Malicious"],
            patch_artist=True,
            boxprops=dict(facecolor="#dfe7f5"),
        )
        ax.set_title(feature, fontsize=9)

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature Distributions by Class", y=1.02)
    return _save_fig(fig, filename)


def plot_pairplot(
    feature_df: pd.DataFrame,
    features: Sequence[str],
    label_col: str = "label",
    filename: str = "pairplot.png",
) -> str:
    """Pairwise scatter matrix for a small subset of features, colored by class.

    Restricted to a handful of features by design: a full ~25-feature pair
    plot is unreadable and computationally wasteful. Callers should pass
    the top 4-6 features (e.g. from `plot_feature_importance_preview`).
    """
    if len(features) > 6:
        logger.warning(
            "plot_pairplot received %d features; truncating to first 6 for readability.",
            len(features),
        )
        features = list(features)[:6]

    colors = feature_df[label_col].map({0: _SAFE_COLOR, 1: _MALICIOUS_COLOR})
    axes = scatter_matrix(
        feature_df[list(features)],
        c=colors,
        figsize=(2.2 * len(features), 2.2 * len(features)),
        diagonal="hist",
        alpha=0.5,
    )
    fig = axes[0, 0].get_figure()
    fig.suptitle("Pair Plot (Blue = Safe, Red = Malicious)", y=1.02)
    return _save_fig(fig, filename)


def save_summary_statistics(
    feature_df: pd.DataFrame, filename: str = "summary_statistics.csv"
) -> str:
    """Save descriptive summary statistics (count, mean, std, quartiles, etc.) to CSV."""
    numeric_df = feature_df.select_dtypes(include=[np.number])
    summary = numeric_df.describe().transpose()
    path = METRICS_DIR / filename
    summary.to_csv(path)
    logger.info("Saved summary statistics: %s", path)
    return str(path)


# ------------------------------------------------------------------------
# Model evaluation plots
# ------------------------------------------------------------------------
def plot_confusion_matrix_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: Optional[str] = None,
) -> str:
    """Confusion matrix heatmap for a single model."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Malicious"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    filename = filename or f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    return _save_fig(fig, filename)


def plot_roc_curves(
    results: dict[str, dict], filename: str = "roc_curves_comparison.png"
) -> str:
    """Overlaid ROC curves for multiple models.

    Args:
        results: Dict mapping model_name -> dict with keys 'y_true' and
            'y_proba' (predicted probability of the positive/Malicious class).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, data in results.items():
        RocCurveDisplay.from_predictions(
            data["y_true"], data["y_proba"], name=model_name, ax=ax
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right", fontsize=8)
    return _save_fig(fig, filename)


def plot_precision_recall_curves(
    results: dict[str, dict], filename: str = "precision_recall_curves_comparison.png"
) -> str:
    """Overlaid Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, data in results.items():
        PrecisionRecallDisplay.from_predictions(
            data["y_true"], data["y_proba"], name=model_name, ax=ax
        )
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="lower left", fontsize=8)
    return _save_fig(fig, filename)


def plot_model_comparison_table(
    comparison_df: pd.DataFrame,
    metrics: Sequence[str] = ("accuracy", "precision", "recall", "f1", "roc_auc"),
    filename: str = "model_comparison.png",
) -> str:
    """Grouped bar chart comparing metrics across all trained models.

    Args:
        comparison_df: DataFrame indexed/columned so that rows = models and
            columns include at least the given `metrics`.
        metrics: Which metric columns to plot.
    """
    n_models = len(comparison_df)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[metric].values, width, label=metric)

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(comparison_df.index, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison Across Metrics")
    ax.legend(loc="lower right", fontsize=8)
    return _save_fig(fig, filename)
