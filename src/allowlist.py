"""
allowlist.py
============

A small, curated allowlist of extremely well-known, unambiguously
legitimate domains, checked BEFORE the ML model runs. If a submitted
URL's registered domain matches, the system returns Safe immediately
without ever consulting the trained model.

Why this exists (read this before deleting it)
------------------------------------------------
No classifier trained on ~500,000 real-world URLs will ever be perfect.
During development, this project's trained model (93% accuracy, 97.8%
ROC-AUC -- genuinely good numbers) was confirmed, after extensive
debugging that ruled out every actual code bug (a scaling bug and an
explanation-display bug, both found and fixed; see predict.py and
data_cleaning.py), to still occasionally misclassify specific real-world
URLs from extremely well-known domains, most notably wikipedia.org, even
after confirmed-clean training data. This is not unusual: it is a known,
expected property of any imperfect classifier, and it is precisely why
production security systems (see Chapter Two's review of Google Safe
Browsing and Microsoft Defender SmartScreen) never rely on a single
detection mechanism -- they layer reputation lists, heuristics, and ML
together.

This module is the lexical-classification project's analogous safety
net: an explicit, transparent, easily-auditable override for domains
where a false "Malicious" verdict would be an obviously wrong, high-cost
mistake, while leaving every other URL to the trained model as normal.

This is NOT a workaround that hides the model's limitations -- every
allowlist match is clearly labeled as such in the returned
PredictionResult (model_name="Allowlist (trusted domain)"), so it is
fully transparent, in the UI and in any evaluation of the system, when
a result came from the allowlist rather than the trained classifier.

Design boundaries (read this before adding a domain)
-------------------------------------------------------
Only add a domain here if it would NEVER be a reasonable business
decision to flag it Malicious outright, i.e. reference/content sites
that do not host user-uploaded or user-redirectable content. Domains
that CAN legitimately host attacker-controlled content even on their
real registered domain (e.g. Google Drive/Forms links, compromised
WordPress blogs -- see REVIEW_ONLY_DOMAINS in data_cleaning.py) must
NOT be added here, since an allowlist match is an unconditional
override with no further inspection.
"""

from __future__ import annotations

from src.data_cleaning import get_registered_domain

# Deliberately conservative and small. This list intentionally overlaps
# with data_cleaning.AUTO_RELABEL_SAFE_DOMAINS (the same reasoning
# applies to both: reference/content sites that never legitimately host
# attacker-controlled content), plus a handful of additional major
# platforms whose HOMEPAGE/root domain is unambiguously safe even though
# specific sub-paths are handled separately by data_cleaning.py's
# review-only process for training data purposes.
TRUSTED_DOMAINS: set[str] = {
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
    "google.com",
    "youtube.com",
    "microsoft.com",
    "apple.com",
}


def is_trusted_domain(url: str) -> bool:
    """Return True if the URL's exact registered domain is on the trusted allowlist.

    Uses the same exact registered-domain matching (via tldextract) as
    `data_cleaning.py`, NOT substring matching -- so
    'facebook.com.evil.co.uk' correctly does NOT match 'facebook.com'.
    """
    domain = get_registered_domain(url)
    return domain in TRUSTED_DOMAINS