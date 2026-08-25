"""
diagnose_label_noise.py — run from your project root: python diagnose_label_noise.py

Checks a curated list of extremely well-known, unambiguously legitimate
domains against the labels actually present in your training data, to
quantify how much real label noise exists (vs. this being a one-off
w3schools.com anomaly).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.utils import PROCESSED_DATA_DIR

# A conservative list: real, extremely well-known, unambiguously legitimate
# domains that should essentially never be labeled Malicious.
KNOWN_SAFE_DOMAINS = [
    "wikipedia.org", "google.com", "youtube.com", "facebook.com", "amazon.com",
    "twitter.com", "instagram.com", "github.com", "microsoft.com", "apple.com",
    "linkedin.com", "reddit.com", "netflix.com", "w3schools.com", "stackoverflow.com",
    "nytimes.com", "bbc.co.uk", "cnn.com", "wordpress.com", "yahoo.com",
    "office.com", "adobe.com", "mozilla.org", "python.org", "npmjs.com",
]

train_path = PROCESSED_DATA_DIR / "train.csv"
test_path = PROCESSED_DATA_DIR / "test.csv"

frames = []
for p in (train_path, test_path):
    if p.exists():
        frames.append(pd.read_csv(p))
df = pd.concat(frames, ignore_index=True)
print(f"Checked across {len(df)} total labeled rows (train + test)\n")

total_matches = 0
total_mislabeled = 0
for domain in KNOWN_SAFE_DOMAINS:
    matches = df[df["url"].str.contains(domain, case=False, na=False, regex=False)]
    if len(matches) == 0:
        continue
    total_matches += len(matches)
    mislabeled = matches[matches["label"] == 1]
    total_mislabeled += len(mislabeled)
    status = "OK" if len(mislabeled) == 0 else f"!! {len(mislabeled)} MISLABELED AS MALICIOUS"
    print(f"{domain:22s} {len(matches):5d} occurrences   {status}")
    if len(mislabeled) > 0:
        for u in mislabeled["url"].head(3):
            print(f"      e.g. {u}")

print()
print(f"Total occurrences of known-safe domains: {total_matches}")
print(f"Total mislabeled as Malicious:           {total_mislabeled}  "
      f"({total_mislabeled/total_matches:.1%} of these known-safe occurrences)" if total_matches else "")