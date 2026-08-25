"""
diagnose.py — run this from your project root: python diagnose.py

Prints everything needed to distinguish between:
  (a) residual class imbalance (technically 2 classes, but still lopsided)
  (b) a feature-extraction quirk specific to bare/scheme-less URLs
  (c) a genuine model-calibration issue worth investigating further
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.feature_engineering import extract_features, FEATURE_NAMES
from src.predict import predict_url
from src.utils import MODELS_DIR, PROCESSED_DATA_DIR

print("=" * 70)
print("1. TRAINING DATA CLASS BALANCE")
print("=" * 70)
import pandas as pd
train_path = PROCESSED_DATA_DIR / "train.csv"
if train_path.exists():
    train_df = pd.read_csv(train_path)
    counts = train_df["label"].value_counts().to_dict()
    total = sum(counts.values())
    safe = counts.get(0, 0)
    malicious = counts.get(1, 0)
    print(f"Safe (0):      {safe:,}  ({safe/total:.1%})")
    print(f"Malicious (1): {malicious:,}  ({malicious/total:.1%})")
    if malicious / total > 0.7:
        print(">>> WARNING: training data is still heavily skewed toward Malicious.")
        print(">>> This alone can cause the model to over-predict Malicious broadly.")
else:
    print("train.csv not found at", train_path)

print()
print("=" * 70)
print("2. SELECTED MODEL AND ITS REPORTED METRICS")
print("=" * 70)
meta_path = MODELS_DIR / "best_model_metadata.json"
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"Model: {meta['model_name']}")
    print(f"Uses scaling: {meta['uses_scaling']}")
    print("Test-set metrics:")
    for k, v in meta["metrics"].items():
        print(f"  {k:10s}: {v:.4f}")
else:
    print("best_model_metadata.json not found at", meta_path)

print()
print("=" * 70)
print("3. PREDICTIONS FOR SEVERAL VARIANTS OF THE WIKIPEDIA URL")
print("=" * 70)
test_urls = [
    "wikipedia.org",
    "www.wikipedia.org",
    "https://wikipedia.org",
    "https://www.wikipedia.org",
    "https://www.wikipedia.org/",
    "https://en.wikipedia.org/wiki/Python",
]
for url in test_urls:
    try:
        result = predict_url(url)
        print(f"\n'{url}'")
        print(f"  -> {result.prediction}  (confidence={result.confidence:.4f}, "
              f"P(malicious)={result.probability_malicious:.4f})")
        print("  Top contributing features:")
        for item in result.top_contributing_features:
            print(f"    {item['feature']:26s} value={item['value']!s:>8}  impact={item['impact']:+.4f}")
    except Exception as e:
        print(f"\n'{url}' -> ERROR: {e}")

print()
print("=" * 70)
print("4. RAW EXTRACTED FEATURES FOR https://www.wikipedia.org/")
print("=" * 70)
feats = extract_features("https://www.wikipedia.org/")
for name in FEATURE_NAMES:
    print(f"  {name:26s} {feats[name]}")