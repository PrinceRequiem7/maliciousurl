"""
data_cleaning.py
=================

Corrects a documented, quantified source of label noise in the training
data: a meaningful fraction of URLs whose REGISTERED DOMAIN belongs to
extremely well-known, unambiguously legitimate reference/content sites
(wikipedia.org, github.com, w3schools.com, ...) are labeled Malicious.
This was discovered and quantified via `diagnose_label_noise.py` during
development -- see that script's output for the investigation this module
is a direct response to.

This is deliberately NOT a blanket "anything from a big company is Safe"
fix, because that would itself introduce a different kind of label noise.
Domains are split into two categories, and only one of them is corrected
automatically:

  - AUTO_RELABEL_SAFE_DOMAINS: reference/content domains (encyclopedias,
    documentation sites, Q&A sites, news outlets, code-hosting for
    reading source, etc.) that essentially never legitimately host
    attacker-controlled content on their own registered domain. An exact
    registered-domain match against this list, on a row labeled
    Malicious, is treated as a confirmed label error and corrected to
    Safe.

  - REVIEW_ONLY_DOMAINS: real, well-known company domains that ARE
    documented vectors for hosting or redirecting to malicious content
    even on their genuine registered domain -- e.g. Google Drive/Forms
    links used to host phishing pages or malware payloads, or a
    compromised user-hosted WordPress blog. Rows matching these are
    NOT automatically relabeled, since doing so risks erasing genuinely
    correct Malicious labels. They are instead written to a CSV for
    manual review.

CRITICAL DESIGN DECISION -- exact registered domain, not substring match:
matching is done against the REGISTERED DOMAIN as parsed by tldextract
(domain + public suffix), never a substring search on the raw URL string.
Substring matching is what originally produced misleading results during
development: URLs such as 'facebook.com.triumphhomes.co.uk' or
'safety.apple.com.mntkaklst...review' contain the target domain as a
literal substring while their ACTUAL registered domain is something else
entirely ('triumphhomes.co.uk', 'mntkaklst...'). These are phishing
domains deliberately crafted to look like the real thing and are
correctly labeled Malicious in the source data; naive substring matching
would incorrectly "correct" them. Exact registered-domain comparison,
which correctly resolves 'en.wikipedia.org' to 'wikipedia.org' while
correctly NOT resolving 'facebook.com.triumphhomes.co.uk' to
'facebook.com', avoids this failure mode entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import tldextract

from src.utils import METRICS_DIR, get_logger

logger = get_logger(__name__)

# Offline-configured, matching the same pattern already used in
# feature_engineering.py: disables tldextract's live fetch of a fresh
# public-suffix-list snapshot (which otherwise attempts a network request
# on every call, contradicting this project's no-network-calls design
# principle and adding needless latency/failure risk in restricted
# environments). Uses tldextract's bundled offline snapshot instead.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

# Reference/content domains: encyclopedic, educational, documentation, or
# editorial content. Never legitimately used to host phishing/malware
# payloads on their own registered domain. Safe to auto-correct.
AUTO_RELABEL_SAFE_DOMAINS: set[str] = {
    "wikipedia.org",
    "w3schools.com",
    "github.com",
    "stackoverflow.com",
    "python.org",
    "mozilla.org",
    "nytimes.com",
    "bbc.co.uk",
    "cnn.com",
    "reddit.com",
}

# Well-known company domains that ARE documented vectors for hosting or
# redirecting to malicious content on their genuine registered domain
# (user-generated content, file-sharing/forms features, compromised
# hosted blogs, redirect-through links, etc.). Flagged for manual review
# only -- never auto-relabeled.
REVIEW_ONLY_DOMAINS: set[str] = {
    "google.com",
    "facebook.com",
    "amazon.com",
    "twitter.com",
    "instagram.com",
    "microsoft.com",
    "office.com",
    "apple.com",
    "linkedin.com",
    "netflix.com",
    "yahoo.com",
    "wordpress.com",
    "adobe.com",
    "youtube.com",
}


@dataclass
class CleaningReport:
    """Summary of what this cleaning pass did, for transparency and reproducibility."""

    total_rows: int
    auto_relabeled_count: int
    review_flagged_count: int
    auto_relabeled_by_domain: dict = field(default_factory=dict)
    review_flagged_by_domain: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "auto_relabeled_count": self.auto_relabeled_count,
            "review_flagged_count": self.review_flagged_count,
            "auto_relabeled_by_domain": self.auto_relabeled_by_domain,
            "review_flagged_by_domain": self.review_flagged_by_domain,
        }


def get_registered_domain(url: str) -> str:
    """Return the exact registered domain (domain + public suffix) for a URL.

    Uses tldextract, which correctly distinguishes 'en.wikipedia.org' (a
    genuine subdomain of wikipedia.org, correctly resolving to
    'wikipedia.org') from 'facebook.com.evil.co.uk' (a domain that merely
    CONTAINS 'facebook.com' as a substring but whose actual registered
    domain is 'evil.co.uk'). Returns an empty string if the URL cannot be
    parsed at all, rather than raising, so this is safe to apply across
    an entire column with `.apply()`.
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    candidate = url.strip()
    if not candidate.startswith(("http://", "https://")):
        candidate = "http://" + candidate
    try:
        ext = _TLD_EXTRACTOR(candidate)
        if not ext.domain or not ext.suffix:
            return ""
        return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        return ""


def clean_mislabeled_rows(
    df: pd.DataFrame,
    url_col: str = "url",
    label_col: str = "label",
    save_review_csv: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Correct confirmed label errors and flag ambiguous ones for manual review.

    Args:
        df: DataFrame with a URL column and a binary label column
            (0=Safe, 1=Malicious).
        url_col: Name of the URL column.
        label_col: Name of the binary label column.
        save_review_csv: If True, write REVIEW_ONLY_DOMAINS matches to
            results/metrics/review_flagged_urls.csv for manual inspection.

    Returns:
        Tuple of (corrected DataFrame, CleaningReport). The input
        DataFrame is not modified in place; a corrected copy is returned.
    """
    working = df.copy()
    registered_domains = working[url_col].astype(str).apply(get_registered_domain)

    auto_mask = registered_domains.isin(AUTO_RELABEL_SAFE_DOMAINS) & (working[label_col] == 1)
    review_mask = registered_domains.isin(REVIEW_ONLY_DOMAINS) & (working[label_col] == 1)

    auto_by_domain = registered_domains[auto_mask].value_counts().to_dict()
    review_by_domain = registered_domains[review_mask].value_counts().to_dict()

    if auto_mask.any():
        logger.warning(
            "Correcting %d rows with confirmed label errors (exact registered-domain "
            "match against known-safe reference domains, previously labeled Malicious): %s",
            int(auto_mask.sum()),
            auto_by_domain,
        )
        working.loc[auto_mask, label_col] = 0
    else:
        logger.info("No confirmed label errors found for the auto-relabel domain list.")

    if review_mask.any():
        logger.warning(
            "Flagged %d rows for MANUAL review (registered domain matches a well-known "
            "company that can legitimately host abused content, e.g. Google Drive/Forms "
            "links) -- these were NOT auto-relabeled: %s",
            int(review_mask.sum()),
            review_by_domain,
        )
        if save_review_csv:
            review_path = METRICS_DIR / "review_flagged_urls.csv"
            working.loc[review_mask, [url_col, label_col]].to_csv(review_path, index=False)
            logger.info("Saved %d review candidates to %s", int(review_mask.sum()), review_path)

    report = CleaningReport(
        total_rows=len(working),
        auto_relabeled_count=int(auto_mask.sum()),
        review_flagged_count=int(review_mask.sum()),
        auto_relabeled_by_domain=auto_by_domain,
        review_flagged_by_domain=review_by_domain,
    )
    return working, report