"""
feature_engineering.py
======================

Extracts 25 lexical features from raw URL strings without performing external
WHOIS, DNS, or network lookups.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract

from src.utils import calculate_entropy, contains_ip_address, get_logger

logger = get_logger(__name__)

# Canonical order of 25 lexical features
FEATURE_NAMES: list[str] = [
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_symbols",
    "num_ampersands",
    "num_exclamation_marks",
    "num_digits",
    "num_letters",
    "num_subdomains",
    "has_ip_address",
    "has_https",
    "has_http",
    "url_entropy",
    "domain_entropy",
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "suspicious_keyword_count",
]

SUSPICIOUS_KEYWORDS: set[str] = {
    "login", "verify", "update", "account", "banking", "secure", "ebayisapi",
    "webscr", "signin", "confirm", "admin", "paypal", "password"
}


def normalize_url(raw_url: str) -> str:
    """Validate input type and ensure URL has a scheme so urlparse extracts netloc correctly."""
    if not isinstance(raw_url, str):
        raise ValueError("URL input must be a string.")

    url = raw_url.strip()
    if not url:
        raise ValueError("URL string cannot be empty or whitespace-only.")

    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


# --- Individual Extractor Functions ---

def url_length(raw_url: str) -> int:
    """Return raw length of URL string."""
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("Invalid URL input.")
    return len(raw_url)


def num_dots(raw_url: str) -> int:
    """Return count of dots in URL."""
    return raw_url.count(".")


def has_https(raw_url: str) -> int:
    """Return 1 if scheme is HTTPS, 0 otherwise."""
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("Invalid URL input.")
    url = raw_url.strip()
    return 1 if url.lower().startswith("https://") else 0


def has_ip_address(raw_url: str) -> int:
    """Return 1 if URL contains an IPv4 or IPv6 address, 0 otherwise."""
    return 1 if contains_ip_address(raw_url) else 0


def num_subdomains(raw_url: str) -> int:
    """Return count of subdomains."""
    url = normalize_url(raw_url)
    ext = tldextract.extract(url)
    return len(ext.subdomain.split(".")) if ext.subdomain else 0


def num_suspicious_keywords(raw_url: str) -> int:
    """Return count of suspicious keywords found in URL."""
    url_lower = raw_url.lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)


def path_length(raw_url: str) -> int:
    """Return length of path component."""
    url = normalize_url(raw_url)
    parsed = urlparse(url)
    return len(parsed.path)


def query_length(raw_url: str) -> int:
    """Return length of query string component."""
    url = normalize_url(raw_url)
    parsed = urlparse(url)
    return len(parsed.query)


# --- Core Feature Extraction ---

def extract_features(raw_url: str) -> dict[str, Any]:
    """Extract 25 lexical features from a single raw URL string."""
    url = normalize_url(raw_url)

    # Robust URL parsing: handle malformed IPv6 brackets in dataset URLs
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)
    except ValueError:
        # Fallback for URLs with unencoded '[' or ']' causing ValueError in urlparse
        safe_url = url.replace("[", "%5B").replace("]", "%5D")
        try:
            parsed = urlparse(safe_url)
            ext = tldextract.extract(safe_url)
        except ValueError:
            parsed = urlparse("http://invalid-malformed-url.local")
            ext = tldextract.extract("http://invalid-malformed-url.local")

    registered_domain = (
        ext.top_domain_under_public_suffix
        if ext.top_domain_under_public_suffix
        else parsed.netloc
    )
    subdomain_part = ext.subdomain

    num_subdomains_val = len(subdomain_part.split(".")) if subdomain_part else 0
    url_len = len(url)

    # Count character types
    num_digits = sum(c.isdigit() for c in url)
    num_letters = sum(c.isalpha() for c in url)
    num_special = url_len - (num_digits + num_letters)

    kw_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower())

    return {
        "url_length": url_len,
        "domain_length": len(registered_domain),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_at_symbols": url.count("@"),
        "num_ampersands": url.count("&"),
        "num_exclamation_marks": url.count("!"),
        "num_digits": num_digits,
        "num_letters": num_letters,
        "num_subdomains": num_subdomains_val,
        "has_ip_address": 1 if contains_ip_address(url) else 0,
        "has_https": 1 if parsed.scheme == "https" else 0,
        "has_http": 1 if parsed.scheme == "http" else 0,
        "url_entropy": calculate_entropy(url),
        "domain_entropy": calculate_entropy(registered_domain),
        "digit_ratio": num_digits / url_len if url_len > 0 else 0.0,
        "letter_ratio": num_letters / url_len if url_len > 0 else 0.0,
        "special_char_ratio": num_special / url_len if url_len > 0 else 0.0,
        "suspicious_keyword_count": kw_count,
    }


def build_feature_matrix(
    df: pd.DataFrame, url_col: str = "url", show_progress: bool = False
) -> pd.DataFrame:
    """Extract features for a full DataFrame containing a URL column."""
    if url_col not in df.columns:
        raise ValueError(f"Column '{url_col}' not found in DataFrame.")

    features_list = [extract_features(url) for url in df[url_col]]
    feature_df = pd.DataFrame(features_list, index=df.index)[FEATURE_NAMES]
    if "label" in df.columns:
        feature_df["label"] = df["label"].values
    return feature_df