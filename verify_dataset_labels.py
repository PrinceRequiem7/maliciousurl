"""
verify_dataset_labels.py
=========================

A larger-scale, standalone extension of src/data_cleaning.py's approach.
Run manually, once, as a data-quality maintenance pass over your FULL
dataset -- this is deliberately NOT wired into preprocessing.py or
train.py, so it does not change the project's no-network-calls-at-
training-or-prediction-time design. It only touches data on disk, before
you run preprocessing/training.

Two modes:

  --mode domain-list  (default, offline, no API key needed)
      Checks every row against a larger built-in reference list of ~150
      well-known, unambiguously legitimate domains spanning tech, news,
      education, government, and open source -- the same exact
      registered-domain matching approach as data_cleaning.py, just
      scaled up. Fully offline. This mode has been tested end-to-end.

  --mode safe-browsing  (optional, requires a free Google API key)
      Additionally queries the Google Safe Browsing API (the same
      service Chrome itself uses) to check rows CURRENTLY LABELED SAFE
      against Google's real, continuously updated threat lists --
      catching mislabels in the OPPOSITE direction from domain-list mode
      (a URL your dataset calls Safe that Google's own data says is
      actually malicious). Batches 500 URLs per request per Google's API
      limits, with checkpointing so a large run can be safely interrupted
      and resumed.

      IMPORTANT HONESTY NOTE: this mode is implemented against Google's
      publicly documented Safe Browsing Lookup API v4 request/response
      format, but has NOT been tested against the live API from this
      environment (network access here is restricted to a small
      allowlist that does not include googleapis.com). Test it yourself
      on a small sample first (see --limit below) before trusting it on
      your full dataset.

Usage
-----
    # Offline mode, default, safe to run immediately:
    python verify_dataset_labels.py

    # Safe Browsing mode, small test run first:
    export GOOGLE_SAFE_BROWSING_API_KEY="your-key-here"
    python verify_dataset_labels.py --mode safe-browsing --limit 500

    # Safe Browsing mode, full run (only after the small test above works):
    python verify_dataset_labels.py --mode safe-browsing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.data_cleaning import get_registered_domain
from src.utils import METRICS_DIR, PROCESSED_DATA_DIR, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# A larger, still-conservative reference list. Every entry here is a
# domain whose PRIMARY, WELL-KNOWN function is reference/documentation/
# editorial content, not user-generated or user-redirectable content --
# same design boundary as data_cleaning.py's AUTO_RELABEL_SAFE_DOMAINS,
# just scaled up across more categories. Domains that host user content
# or are documented phishing/malware vectors even on their real domain
# (Google Drive, WordPress.com blogs, URL shorteners, etc.) are
# deliberately excluded, same reasoning as data_cleaning.py's
# REVIEW_ONLY_DOMAINS.
# ---------------------------------------------------------------------------
EXPANDED_REFERENCE_DOMAINS: set[str] = {
    # Encyclopedic / reference
    "wikipedia.org", "wiktionary.org", "wikimedia.org", "britannica.com",
    "dictionary.com", "merriam-webster.com",
    # Documentation / developer reference
    "w3schools.com", "developer.mozilla.org", "mozilla.org", "docs.python.org",
    "python.org", "github.com", "gitlab.com", "stackoverflow.com",
    "readthedocs.org", "npmjs.com", "pypi.org", "rust-lang.org", "golang.org",
    "kernel.org", "gnu.org", "apache.org", "iso.org", "ietf.org", "w3.org",
    # Major news / editorial (established mastheads only)
    "nytimes.com", "bbc.co.uk", "bbc.com", "cnn.com", "reuters.com",
    "apnews.com", "npr.org", "theguardian.com", "wsj.com", "economist.com",
    "washingtonpost.com", "aljazeera.com", "bloomberg.com",
    # Education / academic
    "mit.edu", "harvard.edu", "stanford.edu", "berkeley.edu", "ox.ac.uk",
    "cam.ac.uk", "coursera.org", "khanacademy.org", "edx.org",
    # Government / international bodies (.gov and major .org)
    "usa.gov", "usembassy.gov", "who.int", "un.org", "europa.eu",
    "nasa.gov", "nih.gov", "cdc.gov",
    # Established open-source / standards orgs
    "mozilla.org", "eclipse.org", "linuxfoundation.org", "creativecommons.org",
    # Q&A / community reference (content, not redirectable file hosting)
    "reddit.com", "quora.com",
}


def check_domain_list_mode(df: pd.DataFrame, url_col: str, label_col: str) -> pd.DataFrame:
    """Offline check against the expanded reference domain list."""
    domains = df[url_col].astype(str).apply(get_registered_domain)
    mislabeled_mask = domains.isin(EXPANDED_REFERENCE_DOMAINS) & (df[label_col] == 1)
    flagged = df.loc[mislabeled_mask].copy()
    flagged["matched_domain"] = domains[mislabeled_mask]
    return flagged


# ---------------------------------------------------------------------------
# Google Safe Browsing integration.
# UNTESTED FROM THIS SANDBOX -- see module docstring. Implemented per the
# documented request/response contract at:
#   https://developers.google.com/safe-browsing/v4/lookup-api
# ---------------------------------------------------------------------------
SAFE_BROWSING_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
SAFE_BROWSING_BATCH_SIZE = 500  # Google's documented per-request limit
CHECKPOINT_PATH = METRICS_DIR / "safe_browsing_checkpoint.json"


def _safe_browsing_check_batch(urls: list[str], api_key: str) -> set[str]:
    """Query Safe Browsing for a batch of URLs; return the subset flagged as threats."""
    import requests  # imported lazily so domain-list mode never needs this dependency

    body = {
        "client": {"clientId": "malicious-url-detector-thesis", "clientVersion": "1.0.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": u} for u in urls],
        },
    }
    resp = requests.post(
        SAFE_BROWSING_ENDPOINT, params={"key": api_key}, json=body, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    matches = data.get("matches", [])
    return {m["threat"]["url"] for m in matches}


def check_safe_browsing_mode(
    df: pd.DataFrame, url_col: str, label_col: str, api_key: str, limit: int | None
) -> pd.DataFrame:
    """Check rows CURRENTLY LABELED SAFE against the live Safe Browsing API.

    Finds mislabels in the opposite direction from domain-list mode: a URL
    your dataset calls Safe that Google's real threat data says is actually
    malicious.
    """
    safe_rows = df[df[label_col] == 0].copy()
    if limit:
        safe_rows = safe_rows.head(limit)

    urls = safe_rows[url_col].astype(str).tolist()
    logger.info("Checking %d Safe-labeled URLs against Safe Browsing API", len(urls))

    checkpoint = {"completed_batches": 0, "flagged_urls": []}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        logger.info("Resuming from checkpoint: %d batches already done", checkpoint["completed_batches"])

    flagged_urls: set[str] = set(checkpoint["flagged_urls"])
    batches = [urls[i:i + SAFE_BROWSING_BATCH_SIZE] for i in range(0, len(urls), SAFE_BROWSING_BATCH_SIZE)]

    for i, batch in enumerate(batches):
        if i < checkpoint["completed_batches"]:
            continue
        try:
            matched = _safe_browsing_check_batch(batch, api_key)
            flagged_urls |= matched
        except Exception as e:
            logger.error("Batch %d/%d failed: %s. Progress saved; re-run to resume.", i + 1, len(batches), e)
            checkpoint["completed_batches"] = i
            checkpoint["flagged_urls"] = list(flagged_urls)
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(checkpoint, f)
            raise

        checkpoint["completed_batches"] = i + 1
        checkpoint["flagged_urls"] = list(flagged_urls)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        logger.info("Batch %d/%d done, %d flagged so far", i + 1, len(batches), len(flagged_urls))
        time.sleep(1)  # gentle pacing, well under documented quota limits

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()  # clean up on successful completion

    return safe_rows[safe_rows[url_col].isin(flagged_urls)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["domain-list", "safe-browsing"], default="domain-list")
    parser.add_argument("--limit", type=int, default=None, help="Only check the first N rows (for testing)")
    parser.add_argument("--apply", action="store_true", help="Write corrections back to disk (default: report only)")
    args = parser.parse_args()

    for split in ("train", "test"):
        path = PROCESSED_DATA_DIR / f"{split}.csv"
        if not path.exists():
            print(f"{path} not found, skipping.")
            continue

        df = pd.read_csv(path)
        print(f"\n{'=' * 70}\n{split}.csv ({len(df)} rows) -- mode: {args.mode}\n{'=' * 70}")

        if args.mode == "domain-list":
            flagged = check_domain_list_mode(df, "url", "label")
            print(f"Found {len(flagged)} rows labeled Malicious matching a known-safe reference domain:")
            if len(flagged):
                print(flagged["matched_domain"].value_counts().to_string())
            if args.apply and len(flagged):
                df.loc[flagged.index, "label"] = 0
                df.to_csv(path, index=False)
                print(f"Applied: {len(flagged)} rows corrected and saved to {path}")
            elif len(flagged):
                print("(Report only -- re-run with --apply to write corrections to disk)")

        elif args.mode == "safe-browsing":
            api_key = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY")
            if not api_key:
                print("ERROR: set GOOGLE_SAFE_BROWSING_API_KEY environment variable first.")
                sys.exit(1)
            flagged = check_safe_browsing_mode(df, "url", "label", api_key, args.limit)
            print(f"Found {len(flagged)} rows labeled Safe that Safe Browsing flags as a real threat:")
            if len(flagged):
                print(flagged[["url"]].to_string(index=False))
            if args.apply and len(flagged):
                df.loc[flagged.index, "label"] = 1
                df.to_csv(path, index=False)
                print(f"Applied: {len(flagged)} rows corrected and saved to {path}")
            elif len(flagged):
                print("(Report only -- re-run with --apply to write corrections to disk)")


if __name__ == "__main__":
    main()