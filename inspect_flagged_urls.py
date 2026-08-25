"""
inspect_flagged_urls.py — run from your project root: python inspect_flagged_urls.py

Shows the ACTUAL URLs flagged by verify_dataset_labels.py's domain-list
mode, not just the per-domain counts -- so you can visually sanity-check
before running --apply. Prints up to 15 examples per domain.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.data_cleaning import get_registered_domain
from src.utils import PROCESSED_DATA_DIR
from verify_dataset_labels import EXPANDED_REFERENCE_DOMAINS

df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
domains = df["url"].astype(str).apply(get_registered_domain)

# Focus on the biggest ones first, especially the outlier
domains_to_check = domains[domains.isin(EXPANDED_REFERENCE_DOMAINS)].value_counts()
print("Flagged domains by count (largest first):")
print(domains_to_check.to_string())
print()

for domain in domains_to_check.index[:5]:  # top 5 largest, ietf.org should be #1
    mask = (domains == domain) & (df["label"] == 1)
    print(f"\n{'=' * 70}\n{domain}  ({mask.sum()} flagged rows) -- sample of 15\n{'=' * 70}")
    print(df.loc[mask, "url"].head(15).to_string(index=False))