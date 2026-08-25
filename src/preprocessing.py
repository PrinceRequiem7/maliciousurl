

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import PROCESSED_DATA_DIR, RAW_DATA_DIR, get_logger

logger = get_logger(__name__)

# Original Kaggle dataset labels -> binary target.
# 0 = Safe, 1 = Malicious.
LABEL_MAP: dict[str, int] = {
    "benign": 0,
    "phishing": 1,
    "malware": 1,
    "defacement": 1,
}

RAW_URL_COLUMN = "url"
RAW_LABEL_COLUMN = "type"
BINARY_LABEL_COLUMN = "label"  # 0 = Safe, 1 = Malicious


@dataclass
class PreprocessingReport:
    """Summary statistics produced during preprocessing.

    Kept as a dataclass (rather than a bare dict) so downstream code and
    the dissertation report-generation step get type-checked, discoverable
    fields instead of loosely-keyed dictionary access.
    """

    raw_row_count: int
    duplicate_rows_removed: int
    missing_value_rows_removed: int
    unmapped_label_rows_removed: int
    final_row_count: int
    class_balance: dict[str, int]
    train_row_count: int
    test_row_count: int

    def to_dict(self) -> dict:
        """Convert the report to a plain dict (e.g. for JSON serialization)."""
        return {
            "raw_row_count": self.raw_row_count,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "missing_value_rows_removed": self.missing_value_rows_removed,
            "unmapped_label_rows_removed": self.unmapped_label_rows_removed,
            "final_row_count": self.final_row_count,
            "class_balance": self.class_balance,
            "train_row_count": self.train_row_count,
            "test_row_count": self.test_row_count,
        }


def load_raw_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw Kaggle Malicious URLs dataset from disk.

    Args:
        csv_path: Path to the raw CSV file. Defaults to
            ``data/raw/malicious_urls_dataset.csv`` if not provided.

    Returns:
        A DataFrame with (at least) the raw ``url`` and ``type`` columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
        ValueError: If the expected columns are not present in the file.
    """
    if csv_path is None:
        csv_path = RAW_DATA_DIR / "malicious_urls_dataset.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{csv_path}'. Download the Kaggle "
            "'Malicious URLs Dataset' and place it at this path, or pass "
            "an explicit csv_path to load_raw_dataset()."
        )

    logger.info("Loading raw dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    missing_columns = {RAW_URL_COLUMN, RAW_LABEL_COLUMN} - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Raw dataset is missing expected column(s): {missing_columns}. "
            f"Found columns: {list(df.columns)}. If the Kaggle dataset "
            "uses different column names, rename them to 'url' and 'type' "
            "before loading, and document this adaptation."
        )

    logger.info("Loaded %d raw rows", len(df))
    return df


def clean_and_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Clean the raw dataset and map labels to a binary target.

    Steps: unmapped-label removal, missing-URL removal, duplicate removal,
    binary label encoding. Order matters — see module docstring for the
    reasoning behind each step's placement.

    Args:
        df: Raw DataFrame with ``url`` and ``type`` columns.

    Returns:
        A tuple of (cleaned DataFrame with columns ``url`` and ``label``,
        a PreprocessingReport summarizing what was removed and why).

    Raises:
        ValueError: If the input DataFrame is empty or missing required
            columns.
    """
    if df.empty:
        raise ValueError("Cannot preprocess an empty DataFrame.")
    missing_columns = {RAW_URL_COLUMN, RAW_LABEL_COLUMN} - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame missing required column(s): {missing_columns}")

    raw_row_count = len(df)
    working = df[[RAW_URL_COLUMN, RAW_LABEL_COLUMN]].copy()

    # --- Step 1: drop rows whose label doesn't map to Safe/Malicious ---
    # CRITICAL: match against a NORMALIZED (stripped, lowercased) copy of the
    # label column, not the raw text. Downloaded CSVs frequently have
    # inconsistent label casing or stray whitespace (e.g. "Benign", "benign ",
    # "BENIGN"). Matching byte-for-byte against LABEL_MAP without normalizing
    # first would silently drop every row whose label doesn't match exactly,
    # which can wipe out an entire class (most dangerously the Safe class)
    # without ever raising an error -- the pipeline would "succeed" while
    # training on badly skewed or effectively single-class data, producing a
    # model that predicts one label almost regardless of input.
    known_label_mask = (
        working[RAW_LABEL_COLUMN].astype(str).str.strip().str.lower().isin(LABEL_MAP.keys())
    )
    unmapped_label_rows_removed = int((~known_label_mask).sum())
    if unmapped_label_rows_removed:
        logger.warning(
            "Removing %d rows with unrecognized label values: %s",
            unmapped_label_rows_removed,
            sorted(set(working.loc[~known_label_mask, RAW_LABEL_COLUMN].astype(str))),
        )
    working = working.loc[known_label_mask].reset_index(drop=True)

    # --- Step 2: drop rows with missing/empty URL ---
    # See module docstring: no sound way to impute a missing URL string.
    url_is_present = working[RAW_URL_COLUMN].notna() & (
        working[RAW_URL_COLUMN].astype(str).str.strip() != ""
    )
    missing_value_rows_removed = int((~url_is_present).sum())
    working = working.loc[url_is_present].reset_index(drop=True)

    # --- Step 3: remove duplicate URLs (before splitting, to prevent leakage) ---
    before_dedup = len(working)
    working = working.drop_duplicates(subset=[RAW_URL_COLUMN]).reset_index(drop=True)
    duplicate_rows_removed = before_dedup - len(working)

    # --- Step 4: encode binary label ---
    # Re-normalize from 'working' here (rather than reusing an earlier
    # intermediate Series) so this is correct regardless of how many rows
    # were subsequently dropped in Steps 2-3 -- no risk of index/length drift.
    normalized_labels = working[RAW_LABEL_COLUMN].astype(str).str.strip().str.lower()
    working[BINARY_LABEL_COLUMN] = normalized_labels.map(LABEL_MAP).astype(int)
    working = working[[RAW_URL_COLUMN, BINARY_LABEL_COLUMN]].rename(
        columns={RAW_URL_COLUMN: "url"}
    )

    class_balance = working[BINARY_LABEL_COLUMN].value_counts().to_dict()
    class_balance = {
        ("Safe" if k == 0 else "Malicious"): int(v) for k, v in class_balance.items()
    }

    # --- Step 5: hard safety check on class balance ---
    # CRITICAL: this is the guard against the exact failure mode that motivated
    # this fix. If label matching (Step 1) ever silently drops most or all of
    # one class -- due to a casing mismatch, an unexpected label spelling, a
    # wrong column, etc. -- training would otherwise proceed "successfully" on
    # badly skewed or effectively single-class data, producing a model that
    # predicts one label almost regardless of input (e.g. flagging
    # unambiguously safe sites like wikipedia.org as Malicious). Rather than
    # letting that fail silently and only surface as bad predictions much
    # later, this raises immediately, loudly, and with an actionable message.
    safe_count = class_balance.get("Safe", 0)
    malicious_count = class_balance.get("Malicious", 0)
    total = safe_count + malicious_count
    MIN_CLASS_FRACTION = 0.05  # each class must be at least 5% of the cleaned data

    if total == 0:
        raise ValueError(
            "Preprocessing produced zero labeled rows after cleaning. "
            f"Raw label values found in the 'type' column before matching: "
            f"{sorted(set(df[RAW_LABEL_COLUMN].astype(str)))[:20]}. "
            f"Expected (case-insensitive): {sorted(LABEL_MAP.keys())}. "
            "Check that your raw CSV's label column actually contains these values."
        )

    safe_fraction = safe_count / total
    malicious_fraction = malicious_count / total
    if safe_fraction < MIN_CLASS_FRACTION or malicious_fraction < MIN_CLASS_FRACTION:
        raise ValueError(
            f"Class balance after cleaning is dangerously skewed: "
            f"Safe={safe_count} ({safe_fraction:.1%}), "
            f"Malicious={malicious_count} ({malicious_fraction:.1%}). "
            f"A model trained on this would learn to predict almost exclusively "
            f"one label. This usually means most rows of one class were dropped "
            f"during label matching -- check the 'unmapped_label_rows_removed' "
            f"count and the logged list of unrecognized label values above, and "
            f"confirm your raw CSV's 'type' column actually contains the "
            f"expected values: {sorted(LABEL_MAP.keys())}."
        )

    report = PreprocessingReport(
        raw_row_count=raw_row_count,
        duplicate_rows_removed=duplicate_rows_removed,
        missing_value_rows_removed=missing_value_rows_removed,
        unmapped_label_rows_removed=unmapped_label_rows_removed,
        final_row_count=len(working),
        class_balance=class_balance,
        train_row_count=0,  # populated after split
        test_row_count=0,
    )

    logger.info(
        "Cleaning complete: %d -> %d rows (removed %d unmapped-label, "
        "%d missing-URL, %d duplicate). Class balance: %s",
        raw_row_count,
        len(working),
        unmapped_label_rows_removed,
        missing_value_rows_removed,
        duplicate_rows_removed,
        class_balance,
    )

    return working, report


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform an 80/20 stratified train/test split on the binary label.

    Stratification guarantees both splits preserve the overall
    Safe/Malicious ratio (see module docstring for why this matters).

    Args:
        df: Cleaned DataFrame with ``url`` and ``label`` columns.
        test_size: Fraction of data to allocate to the test split.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If required columns are missing or df is too small
            to stratify (fewer than 2 examples per class).
    """
    required = {"url", BINARY_LABEL_COLUMN}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame missing required column(s): {required - set(df.columns)}")

    class_counts = df[BINARY_LABEL_COLUMN].value_counts()
    if (class_counts < 2).any():
        raise ValueError(
            "Cannot perform a stratified split: at least one class has "
            f"fewer than 2 examples. Class counts: {class_counts.to_dict()}"
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[BINARY_LABEL_COLUMN],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Stratified split complete: %d train rows, %d test rows (test_size=%.2f)",
        len(train_df),
        len(test_df),
        test_size,
    )
    return train_df, test_df


def run_preprocessing_pipeline(
    csv_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_outputs: bool = True,
    clean_known_domain_labels: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessingReport]:
    """Run the full preprocessing pipeline end to end.

    Loads the raw dataset, cleans and encodes it, optionally corrects
    confirmed label errors for well-known reference domains (see
    `src/data_cleaning.py`), performs the stratified split, and
    (optionally) writes ``train.csv`` / ``test.csv`` to ``data/processed/``.

    Args:
        csv_path: Path to the raw CSV. Defaults to the standard raw path.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.
        save_outputs: If True, persist train/test CSVs to
            ``data/processed/``.
        clean_known_domain_labels: If True (default), applies
            `data_cleaning.clean_mislabeled_rows` before the train/test
            split, correcting confirmed label errors (e.g. wikipedia.org
            article URLs mislabeled Malicious) and flagging ambiguous
            cases (e.g. Google Drive links) for manual review rather than
            auto-correcting them. Applied BEFORE the split, not after, so
            the same corrected label is consistently reflected in
            whichever split a given row lands in.

    Returns:
        Tuple of (train_df, test_df, PreprocessingReport).
    """
    raw_df = load_raw_dataset(csv_path)
    clean_df, report = clean_and_encode(raw_df)

    if clean_known_domain_labels:
        from src.data_cleaning import clean_mislabeled_rows

        clean_df, cleaning_report = clean_mislabeled_rows(clean_df)
        logger.info("Known-domain label cleaning: %s", cleaning_report.to_dict())

    train_df, test_df = stratified_split(
        clean_df, test_size=test_size, random_state=random_state
    )
    report.train_row_count = len(train_df)
    report.test_row_count = len(test_df)

    if save_outputs:
        train_path = PROCESSED_DATA_DIR / "train.csv"
        test_path = PROCESSED_DATA_DIR / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info("Saved processed splits to %s and %s", train_path, test_path)

    return train_df, test_df, report


if __name__ == "__main__":
    # Allows running `python -m src.preprocessing` as a standalone step.
    _, _, _report = run_preprocessing_pipeline()
    logger.info("Preprocessing report: %s", _report.to_dict())