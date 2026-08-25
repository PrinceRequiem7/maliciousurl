"""
clean_existing_data.py — run from your project root: python clean_existing_data.py

Applies the label-noise correction in src/data_cleaning.py directly to
your EXISTING data/processed/train.csv and test.csv, in place, without
re-running the full raw-CSV preprocessing pipeline (which would take
longer on 640k+ rows and isn't necessary here).

Each split is cleaned independently, using its own rows only, so this
does not reshuffle or move any row between train and test -- it only
corrects labels within whichever split a row already belongs to.

After running this, retrain: python -m src.train
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.data_cleaning import clean_mislabeled_rows
from src.utils import PROCESSED_DATA_DIR

for split_name in ("train", "test"):
    path = PROCESSED_DATA_DIR / f"{split_name}.csv"
    if not path.exists():
        print(f"{path} not found, skipping.")
        continue

    df = pd.read_csv(path)
    print(f"\n{'=' * 70}\nCleaning {split_name}.csv ({len(df)} rows)\n{'=' * 70}")

    # Save review CSV only once (from the train split) to avoid overwriting
    # it with a smaller/different set when the test split is processed next.
    cleaned_df, report = clean_mislabeled_rows(df, save_review_csv=(split_name == "train"))

    print(f"Auto-relabeled (confirmed errors, Malicious -> Safe): {report.auto_relabeled_count}")
    if report.auto_relabeled_by_domain:
        for domain, count in sorted(report.auto_relabeled_by_domain.items(), key=lambda x: -x[1]):
            print(f"    {domain:20s} {count}")

    print(f"Flagged for manual review (NOT auto-relabeled): {report.review_flagged_count}")
    if report.review_flagged_by_domain:
        for domain, count in sorted(report.review_flagged_by_domain.items(), key=lambda x: -x[1]):
            print(f"    {domain:20s} {count}")

    cleaned_df.to_csv(path, index=False)
    print(f"Saved corrected {split_name}.csv")

print("\nDone. Re-run 'python -m src.train' to retrain on the corrected labels.")
print("Review-only candidates (not auto-changed) were saved to "
      "results/metrics/review_flagged_urls.csv for you to inspect manually.")