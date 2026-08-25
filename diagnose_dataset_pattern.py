"""
diagnose_dataset_pattern.py — run from your project root: python diagnose_dataset_pattern.py

Investigates WHY wikipedia.org's specific feature values (short url_length,
short domain_length, few dots, few letters, short path) push the model
toward Malicious. Checks two hypotheses:

  A) Genuine dataset characteristic: malicious URLs in this dataset really
     do tend to be shorter/simpler than benign ones (plausible but worth
     seeing directly rather than assuming).

  B) Fallback-path artifact: extract_features() has a try/except that
     falls back to parsing "http://invalid-malformed-url.local" whenever
     urlparse/tldextract raises ValueError (e.g. on malformed/obfuscated
     URLs). If that fires disproportionately on malicious training rows,
     it would cluster a lot of them around similar short/simple feature
     values -- teaching the model a spurious "short = malicious" pattern
     that has nothing to do with real maliciousness.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.feature_engineering import extract_features, build_feature_matrix
from src.utils import PROCESSED_DATA_DIR

train_path = PROCESSED_DATA_DIR / "train.csv"
df = pd.read_csv(train_path)
print(f"Loaded {len(df)} training rows")

# --- Hypothesis B: how often does the malformed-URL fallback fire? ---
print()
print("=" * 70)
print("B. FALLBACK-PATH ARTIFACT CHECK")
print("=" * 70)

FALLBACK_DOMAIN_LEN = len("invalid-malformed-url.local")  # what the fallback produces

sample = df.sample(min(20000, len(df)), random_state=1)
fallback_count_by_label = {0: 0, 1: 0}
bracket_count_by_label = {0: 0, 1: 0}

for _, row in sample.iterrows():
    url = str(row["url"])
    label = row["label"]
    if "[" in url or "]" in url:
        bracket_count_by_label[label] += 1
    try:
        feats = extract_features(url)
        if feats["domain_length"] == FALLBACK_DOMAIN_LEN:
            fallback_count_by_label[label] += 1
    except Exception:
        pass

n_safe = (sample["label"] == 0).sum()
n_mal = (sample["label"] == 1).sum()
print(f"Sampled {len(sample)} rows ({n_safe} Safe, {n_mal} Malicious)")
print(f"Rows containing '[' or ']':  Safe={bracket_count_by_label[0]}  Malicious={bracket_count_by_label[1]}")
print(f"Rows hitting the fallback domain: Safe={fallback_count_by_label[0]}  Malicious={fallback_count_by_label[1]}")
if fallback_count_by_label[1] > n_mal * 0.02:
    print(">>> Fallback path fires on a NON-TRIVIAL fraction of malicious URLs.")
    print(">>> This could genuinely be teaching the model a spurious pattern.")
else:
    print(">>> Fallback path is rare enough that it's probably not the main driver.")

# --- Hypothesis A: real distribution of the 5 flagged features, by label ---
print()
print("=" * 70)
print("A. FEATURE DISTRIBUTIONS BY LABEL (the 5 features flagged for wikipedia.org)")
print("=" * 70)

feat_sample = df.sample(min(15000, len(df)), random_state=2)
fm = build_feature_matrix(feat_sample, show_progress=False)

wiki_values = {
    "url_length": 26, "domain_length": 13, "num_dots": 2,
    "num_letters": 20, "path_length": 1,
}

for feat_name, wiki_val in wiki_values.items():
    if feat_name not in fm.columns:
        print(f"\n{feat_name}  -- not found in feature matrix, skipping")
        continue
    safe_vals = fm.loc[fm["label"] == 0, feat_name]
    mal_vals = fm.loc[fm["label"] == 1, feat_name]
    print(f"\n{feat_name}  (wikipedia.org value = {wiki_val})")
    print(f"  Safe:      mean={safe_vals.mean():.1f}  median={safe_vals.median():.1f}  "
          f"p10={safe_vals.quantile(.10):.1f}  p90={safe_vals.quantile(.90):.1f}")
    print(f"  Malicious: mean={mal_vals.mean():.1f}  median={mal_vals.median():.1f}  "
          f"p10={mal_vals.quantile(.10):.1f}  p90={mal_vals.quantile(.90):.1f}")
    pct_safe_below = (safe_vals <= wiki_val).mean()
    pct_mal_below = (mal_vals <= wiki_val).mean()
    print(f"  % of Safe URLs with value <= wikipedia's:      {pct_safe_below:.1%}")
    print(f"  % of Malicious URLs with value <= wikipedia's: {pct_mal_below:.1%}")

# --- Nearest neighbors: what does the training data actually look like near wikipedia.org? ---
print()
print("=" * 70)
print("C. NEAREST NEIGHBORS in training data (by these 5 features)")
print("=" * 70)
feat_cols = [c for c in wiki_values.keys() if c in fm.columns]
if len(feat_cols) < len(wiki_values):
    missing = set(wiki_values.keys()) - set(feat_cols)
    print(f"(Note: skipping missing columns for neighbor distance: {missing})")
X = fm[feat_cols].to_numpy(dtype=float)
wiki_vec = np.array([wiki_values[c] for c in feat_cols], dtype=float)
# normalize each column so no single feature (e.g. url_length) dominates distance
std = X.std(axis=0)
std[std == 0] = 1
dist = np.sqrt((((X - wiki_vec) / std) ** 2).sum(axis=1))
nearest_idx = np.argsort(dist)[:15]
neighbors = feat_sample.iloc[nearest_idx][["url", "label"]].copy()
neighbors["label"] = neighbors["label"].map({0: "Safe", 1: "Malicious"})
neighbors["distance"] = dist[nearest_idx]
print(neighbors.to_string(index=False))
label_counts = neighbors["label"].value_counts().to_dict()
print(f"\nOf the 15 nearest neighbors to wikipedia.org's feature vector: {label_counts}")