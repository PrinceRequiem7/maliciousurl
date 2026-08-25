"""
verify_current_state.py — run from your project root: python verify_current_state.py

Checks, in order:
  A) Does the CURRENT train.csv/test.csv still contain wikipedia.org /
     google.com rows labeled Malicious? (Confirms whether cleaning
     actually took effect and survived -- e.g. wasn't wiped out by
     re-running raw preprocessing afterward without cleaning again.)
  B) Is the scaling-fix version of predict.py actually the one running?
  C) Live predict_url() output for both URLs, in full detail.
  D) Model file timestamp vs train.csv timestamp -- was the model
     actually retrained AFTER the data was cleaned, or is it stale?
"""
import sys
from pathlib import Path
import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.data_cleaning import get_registered_domain
from src.utils import MODELS_DIR, PROCESSED_DATA_DIR

print("=" * 70)
print("A. CURRENT train.csv / test.csv: any remaining wikipedia.org / google.com")
print("   rows labeled Malicious?")
print("=" * 70)
for split in ("train", "test"):
    path = PROCESSED_DATA_DIR / f"{split}.csv"
    if not path.exists():
        print(f"  {path} not found")
        continue
    df = pd.read_csv(path)
    domains = df["url"].astype(str).apply(get_registered_domain)
    for target in ("wikipedia.org", "google.com"):
        mask = (domains == target) & (df["label"] == 1)
        total = (domains == target).sum()
        print(f"  {split}.csv: {target:16s} total={total:5d}  still labeled Malicious={mask.sum()}")
        if target == "wikipedia.org" and mask.sum() > 0:
            print(f"      >>> UNEXPECTED: cleaning should have fixed ALL of these.")
        if target == "google.com" and mask.sum() > 0:
            print(f"      (expected -- google.com is REVIEW-ONLY by design, not auto-fixed)")

print()
print("=" * 70)
print("B. Is the scaling-fix predict.py actually in place?")
print("=" * 70)
import inspect
from src import predict as predict_module
source = inspect.getsource(predict_module)
if "uses_scaling" in source:
    print("  FOUND 'uses_scaling' in src/predict.py -- the fix appears to be in place.")
else:
    print("  >>> 'uses_scaling' NOT FOUND in src/predict.py -- the scaling fix from")
    print("  >>> earlier is NOT actually in this file. This is likely still broken.")

print()
print("=" * 70)
print("D. Model freshness: was training re-run AFTER the data was cleaned?")
print("=" * 70)
model_path = MODELS_DIR / "best_model.pkl"
train_path = PROCESSED_DATA_DIR / "train.csv"
if model_path.exists() and train_path.exists():
    model_time = datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
    train_time = datetime.datetime.fromtimestamp(train_path.stat().st_mtime)
    print(f"  train.csv last modified:     {train_time}")
    print(f"  best_model.pkl last modified: {model_time}")
    if model_time < train_time:
        print("  >>> best_model.pkl is OLDER than train.csv.")
        print("  >>> The model was NOT retrained after the data was cleaned.")
        print("  >>> Run: python -m src.train")
    else:
        print("  Model is newer than train.csv -- retraining did happen after cleaning.")

print()
print("=" * 70)
print("C. Live predict_url() output")
print("=" * 70)
try:
    from src.predict import predict_url
    for url in ["https://www.wikipedia.org/", "https://www.google.com/", "google.com"]:
        r = predict_url(url)
        print(f"\n  '{url}'")
        print(f"    -> {r.prediction}  (confidence={r.confidence:.4f}, "
              f"P(malicious)={r.probability_malicious:.4f})")
        for item in r.top_contributing_features[:5]:
            print(f"      {item['feature']:26s} value={item['value']!s:>8}  impact={item['impact']:+.4f}")
except Exception as e:
    print(f"  ERROR running predict_url: {e}")