import time, json
import numpy as np
import pandas as pd
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.feature_engineering import FEATURE_NAMES, build_feature_matrix
from src.utils import METRICS_DIR, FIGURES_DIR

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
with open('models/best_model_metadata.json') as f:
    meta = json.load(f)
uses_scaling = meta['uses_scaling']

train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')
train_fm = build_feature_matrix(train_df, show_progress=False)
test_fm = build_feature_matrix(test_df, show_progress=False)

X_train = train_fm[list(FEATURE_NAMES)]
X_test = test_fm[list(FEATURE_NAMES)].reset_index(drop=True)
y_test = test_fm['label'].reset_index(drop=True)

X_train_input = scaler.transform(X_train) if uses_scaling else X_train.values
X_test_input = scaler.transform(X_test) if uses_scaling else X_test.values
feature_names = list(FEATURE_NAMES)

def predict_fn(x):
    return model.predict_proba(x)

# --- Set up explainers ---
lime_explainer = LimeTabularExplainer(
    X_train_input, feature_names=feature_names, class_names=['Safe', 'Malicious'],
    mode='classification', discretize_continuous=True, random_state=42,
)

background = shap.sample(X_train_input, 100, random_state=42)
shap_explainer = shap.Explainer(predict_fn, background, seed=42)

def _extract_feature_name(lime_condition_str: str, known_features: list[str]) -> str:
    """Extract the actual feature name from a LIME condition string.

    LIME's as_list() output for discretized continuous features can take
    several forms depending on which bin the value falls into:
      "feature_name <= 0.15"
      "feature_name > 0.86"
      "-0.86 < feature_name <= -0.15"   <- feature name in the MIDDLE
    Naive whitespace splitting (taking the first token) silently returns
    the numeric bound instead of the feature name for the third form.
    This matches against the known feature name list directly instead.
    """
    for fname in known_features:
        # word-boundary-safe check: fname must appear as a standalone token
        if fname in lime_condition_str.split():
            return fname
    # fallback: some LIME versions glue name to operator with no space
    for fname in known_features:
        if fname in lime_condition_str:
            return fname
    return lime_condition_str  # last resort, should not normally happen

def lime_top_k(idx, k=5, seed=None):
    exp = lime_explainer.explain_instance(
        X_test_input[idx], predict_fn, num_features=k,
        labels=(1,), num_samples=500,
    )
    return set(_extract_feature_name(feat, feature_names) for feat, _ in exp.as_list(label=1))

def shap_top_k(idx, k=5):
    sv = shap_explainer(X_test_input[idx:idx+1])
    vals = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
    ranked = sorted(zip(feature_names, np.abs(vals)), key=lambda x: -x[1])
    return set(f for f, _ in ranked[:k])

# --- 1. Agreement analysis: LIME vs SHAP top-5 features per instance ---
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_test_input), size=10, replace=False)

agreements = []
lime_times = []
shap_times = []
print("=== LIME vs SHAP Top-5 Feature Agreement ===")
for idx in sample_idx:
    t0 = time.perf_counter()
    lime_feats = lime_top_k(idx)
    lime_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    shap_feats = shap_top_k(idx)
    shap_times.append(time.perf_counter() - t0)

    jaccard = len(lime_feats & shap_feats) / len(lime_feats | shap_feats)
    agreements.append(jaccard)
    print(f"idx={idx:4d}  LIME={sorted(lime_feats)}  SHAP={sorted(shap_feats)}  Jaccard={jaccard:.2f}")

print(f"\nMean Jaccard agreement: {np.mean(agreements):.3f} (std={np.std(agreements):.3f})")
print(f"Mean LIME time per explanation: {np.mean(lime_times):.4f}s")
print(f"Mean SHAP time per explanation: {np.mean(shap_times):.4f}s")

# --- 2. Stability analysis: repeat explanation 5x on same instances, measure consistency ---
print("\n=== Stability Analysis (5 repeated runs per instance) ===")
stability_instances = sample_idx[:3]
lime_stability = []
shap_stability = []

for idx in stability_instances:
    lime_runs = []
    for run in range(5):
        exp = lime_explainer.explain_instance(
            X_test_input[idx], predict_fn, num_features=5, labels=(1,), num_samples=500,
        )
        feats = set(_extract_feature_name(f, feature_names) for f, _ in exp.as_list(label=1))
        lime_runs.append(feats)
    pairwise = [len(lime_runs[i] & lime_runs[j]) / len(lime_runs[i] | lime_runs[j])
                for i in range(5) for j in range(i+1, 5)]
    lime_stability.append(np.mean(pairwise))

    shap_runs = []
    for run in range(5):
        shap_runs.append(shap_top_k(idx))
    pairwise = [len(shap_runs[i] & shap_runs[j]) / len(shap_runs[i] | shap_runs[j])
                for i in range(5) for j in range(i+1, 5)]
    shap_stability.append(np.mean(pairwise))

print(f"LIME avg stability (pairwise Jaccard across 5 runs): {np.mean(lime_stability):.3f}")
print(f"SHAP avg stability (pairwise Jaccard across 5 runs): {np.mean(shap_stability):.3f}")

results = {
    "mean_agreement_jaccard": float(np.mean(agreements)),
    "std_agreement_jaccard": float(np.std(agreements)),
    "mean_lime_time_seconds": float(np.mean(lime_times)),
    "mean_shap_time_seconds": float(np.mean(shap_times)),
    "lime_stability": float(np.mean(lime_stability)),
    "shap_stability": float(np.mean(shap_stability)),
    "per_instance_agreement": agreements,
}
with open(METRICS_DIR / "lime_shap_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/metrics/lime_shap_comparison.json")