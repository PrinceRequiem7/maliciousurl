"""
utils.py
========

Shared, dependency-light utilities used across the malicious_url_detector
project. Every other module in `src/` and `app/` imports from here rather
than re-implementing logging, path resolution, validation, or entropy
calculations.

Contents
--------
- Project-relative path constants (DATA_DIR, MODELS_DIR, RESULTS_DIR, ...)
- `get_logger`: consistent console + file logging factory
- `Timer`: context manager for timing training / tuning steps
- `is_valid_url`: lightweight structural URL validation (no network calls)
- `shannon_entropy` / `calculate_entropy`: string entropy calculation
- `contains_ip_address`: raw IP presence check in URL strings
- `SUSPICIOUS_KEYWORDS`: curated keyword list used in feature engineering
"""

from __future__ import annotations

import logging
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

# ------------------------------------------------------------------------
# Project path constants
# ------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Ensure critical output directories exist at import time
for _dir in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    REPORTS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """Create (or retrieve) a configured logger."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        try:
            file_handler = logging.FileHandler(
                REPORTS_DIR / "project.log", encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            logger.warning(
                "Could not attach file handler for logging; "
                "continuing with console logging only."
            )

    logger.propagate = False
    return logger


# ------------------------------------------------------------------------
# Timing utility
# ------------------------------------------------------------------------
class Timer:
    """Context manager for timing a block of code."""

    def __init__(self, label: str = "operation") -> None:
        self.label = label
        self.start_time: float = 0.0
        self.elapsed_seconds: float = 0.0
        self._logger = get_logger("Timer")

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.elapsed_seconds = time.perf_counter() - self.start_time
        self._logger.info(
            "%s completed in %.3f seconds", self.label, self.elapsed_seconds
        )


# ------------------------------------------------------------------------
# URL validation and IP detection
# ------------------------------------------------------------------------
_DOMAIN_HOST_PATTERN = r"([a-zA-Z0-9](-*[a-zA-Z0-9])*\.)+[a-zA-Z]{2,}"
_IPV4_HOST_PATTERN = r"(\d{1,3}\.){3}\d{1,3}"
_URL_STRUCTURE_PATTERN = re.compile(
    r"^(https?://)?"                                       # optional scheme
    rf"({_DOMAIN_HOST_PATTERN}|{_IPV4_HOST_PATTERN})"       # domain OR raw IPv4 host
    r"(:\d+)?"                                              # optional port
    r"(/[^\s]*)?$"                                          # optional path/query/fragment
)

_IP_ADDRESS_PATTERN = re.compile(
    r'(?:^|\b)(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::\d+)?(?:/|\b)|'  # IPv4
    r'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}'                # IPv6
)


def is_valid_url(url: str) -> bool:
    """Validate that a string is structurally plausible as a URL."""
    if not isinstance(url, str):
        return False
    url = url.strip()
    if not url or len(url) > 2048:
        return False
    return bool(_URL_STRUCTURE_PATTERN.match(url))


def contains_ip_address(url: str) -> bool:
    """Check if the URL contains a raw IPv4 or IPv6 address."""
    if not url or not isinstance(url, str):
        return False
    return bool(_IP_ADDRESS_PATTERN.search(url))


# ------------------------------------------------------------------------
# Shannon entropy
# ------------------------------------------------------------------------
def shannon_entropy(text: str) -> float:
    """Compute the Shannon entropy of a string, in bits."""
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)
    entropy = -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )
    if entropy == 0.0:
        return 0.0
    return float(entropy)


# Alias function name expected by feature_engineering.py
calculate_entropy = shannon_entropy


# ------------------------------------------------------------------------
# Suspicious keyword list
# ------------------------------------------------------------------------
SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "login",
    "signin",
    "sign-in",
    "verify",
    "account",
    "update",
    "secure",
    "banking",
    "confirm",
    "password",
    "pay",
    "webscr",
    "ebayisapi",
    "wp-admin",
    "admin",
    "suspend",
    "invoice",
    "free",
    "bonus",
    "click",
    "urgent",
    "alert",
    "recover",
    "unlock",
    "token",
    "reset",
)


def count_suspicious_keywords(url: str) -> int:
    """Count occurrences of curated suspicious keywords in a URL (case-insensitive)."""
    if not url:
        return 0
    url_lower = url.lower()
    return sum(url_lower.count(keyword) for keyword in SUSPICIOUS_KEYWORDS)