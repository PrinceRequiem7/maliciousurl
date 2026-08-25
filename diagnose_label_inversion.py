"""
diagnose_label_inversion.py — run from your project root: python diagnose_label_inversion.py

Definitively tests whether the live predict_url() pipeline agrees with the
ground-truth labels in your own training data, or whether it's systematically
backwards (the classic "labels got swapped somewhere" bug).

This does NOT require me to see predict.py or preprocessing.py -- it tests
actual, observed behavior against ground truth you already have on disk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from src.predict import predict_url
from src.utils import MODELS_DIR, PROCESSED_DATA_DIR

print("=" * 70)
print("A. model.classes_ ORDER (ground truth for what predict_proba columns mean)")
print("=" * 70)
model = joblib.load(MODELS_DIR / "best_model.pkl")
if hasattr(model, "classes_"):
    print(f"  model.classes_ = {model.classes_}")
    print("  scikit-learn's predict_proba() column order EXACTLY follows this array.")
    print("  If this is [0, 1], column 0 = P(label 0), column 1 = P(label 1).")
    print("  If your label encoding is 0=Safe, 1=Malicious, that's the expected order.")
else:
    print("  Model has no classes_ attribute (unexpected for a classifier).")

print()
print("=" * 70)
print("B. LIVE PIPELINE vs GROUND TRUTH: sampling your own labeled test data")
print("=" * 70)
test_path = PROCESSED_DATA_DIR / "test.csv"
if not test_path.exists():
    test_path = PROCESSED_DATA_DIR / "train.csv"
    print(f"  (test.csv not found, using train.csv instead)")

df = pd.read_csv(test_path)
print(f"  Loaded {len(df)} rows from {test_path}")

N = 40
safe_sample = df[df["label"] == 0].sample(min(N, (df["label"] == 0).sum()), random_state=1)
mal_sample = df[df["label"] == 1].sample(min(N, (df["label"] == 1).sum()), random_state=1)

def run_batch(sample_df, true_label_name):
    correct = 0
    wrong_examples = []
    for _, row in sample_df.iterrows():
        try:
            result = predict_url(row["url"])
        except Exception as e:
            continue
        if result.prediction == true_label_name:
            correct += 1
        else:
            wrong_examples.append((row["url"], result.prediction, result.confidence))
    return correct, len(sample_df), wrong_examples

safe_correct, safe_total, safe_wrong = run_batch(safe_sample, "Safe")
mal_correct, mal_total, mal_wrong = run_batch(mal_sample, "Malicious")

print(f"\n  Known-SAFE URLs (ground truth label=0):")
print(f"    predict_url() agreed with ground truth: {safe_correct}/{safe_total} "
      f"({safe_correct/safe_total:.1%})")
print(f"  Known-MALICIOUS URLs (ground truth label=1):")
print(f"    predict_url() agreed with ground truth: {mal_correct}/{mal_total} "
      f"({mal_correct/mal_total:.1%})")

overall = (safe_correct + mal_correct) / (safe_total + mal_total)
print(f"\n  Overall live-pipeline accuracy on this sample: {overall:.1%}")

print()
print("=" * 70)
print("C. INTERPRETATION")
print("=" * 70)
if overall < 0.35:
    print("  >>> STRONG EVIDENCE OF LABEL INVERSION.")
    print("  >>> The live predict_url() pipeline disagrees with ground truth on")
    print("  >>> most examples -- and if it's disagreeing in a CONSISTENT direction")
    print("  >>> (known-Safe called Malicious, known-Malicious called Safe), the")
    print("  >>> 'Safe'/'Malicious' labels are almost certainly swapped somewhere")
    print("  >>> between training and serving.")
    print(f"  >>> Known-Safe URLs predicted wrong: {len(safe_wrong)}/{safe_total}")
    if safe_wrong:
        print(f"       e.g. {safe_wrong[0][0]!r} -> predicted {safe_wrong[0][1]} "
              f"(confidence {safe_wrong[0][2]:.2f})")
elif overall > 0.85:
    print("  >>> Live pipeline agrees well with ground truth on this sample.")
    print("  >>> This does NOT match the earlier wikipedia.org observation, which")
    print("  >>> suggests the issue is specific to real-world URLs NOT well")
    print("  >>> represented in the training distribution, rather than a systemic")
    print("  >>> label inversion. Worth comparing feature values of misclassified")
    print("  >>> real-world URLs against the training data's typical ranges.")
else:
    print("  >>> Mixed result -- not a clean inversion, but worse than the model's")
    print("  >>> own reported metrics would suggest. Please share this full output.")