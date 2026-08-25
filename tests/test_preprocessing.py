"""
test_preprocessing.py
======================

Unit tests for `src/preprocessing.py`: label mapping, cleaning
(duplicates, missing values, unmapped labels), and the stratified split.
"""

import pandas as pd
import pytest

from src.preprocessing import BINARY_LABEL_COLUMN, clean_and_encode, stratified_split


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "url": [
                "http://safe1.com",
                "http://safe1.com",  # duplicate
                "http://safe2.com",
                None,  # missing
                "",  # empty
                "http://phish.com",
                "http://malware.com",
                "http://defaced.com",
                "http://weird-label.com",  # unmapped label
            ],
            "type": [
                "benign",
                "benign",
                "benign",
                "benign",
                "benign",
                "phishing",
                "malware",
                "defacement",
                "not_a_real_label",
            ],
        }
    )


class TestCleanAndEncode:
    def test_removes_duplicates(self, raw_df):
        clean_df, report = clean_and_encode(raw_df)
        assert clean_df["url"].duplicated().sum() == 0
        assert report.duplicate_rows_removed == 1

    def test_removes_missing_urls(self, raw_df):
        clean_df, report = clean_and_encode(raw_df)
        assert clean_df["url"].isna().sum() == 0
        assert report.missing_value_rows_removed == 2

    def test_removes_unmapped_labels(self, raw_df):
        clean_df, report = clean_and_encode(raw_df)
        assert "http://weird-label.com" not in clean_df["url"].values
        assert report.unmapped_label_rows_removed == 1

    def test_binary_label_encoding(self, raw_df):
        clean_df, _ = clean_and_encode(raw_df)
        mapping = dict(zip(clean_df["url"], clean_df[BINARY_LABEL_COLUMN]))
        assert mapping["http://safe1.com"] == 0
        assert mapping["http://safe2.com"] == 0
        assert mapping["http://phish.com"] == 1
        assert mapping["http://malware.com"] == 1
        assert mapping["http://defaced.com"] == 1

    def test_final_row_count(self, raw_df):
        clean_df, report = clean_and_encode(raw_df)
        # 9 raw - 1 unmapped - 2 missing - 1 duplicate = 5
        assert len(clean_df) == 5
        assert report.final_row_count == 5

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError):
            clean_and_encode(pd.DataFrame())

    def test_missing_required_column_raises(self):
        with pytest.raises(ValueError):
            clean_and_encode(pd.DataFrame({"url": ["http://a.com"]}))


class TestStratifiedSplit:
    def test_preserves_class_ratio(self):
        df = pd.DataFrame(
            {
                "url": [f"http://safe{i}.com" for i in range(40)]
                + [f"http://bad{i}.com" for i in range(10)],
                BINARY_LABEL_COLUMN: [0] * 40 + [1] * 10,
            }
        )
        train_df, test_df = stratified_split(df, test_size=0.2, random_state=1)

        assert len(train_df) + len(test_df) == len(df)
        train_ratio = train_df[BINARY_LABEL_COLUMN].mean()
        test_ratio = test_df[BINARY_LABEL_COLUMN].mean()
        overall_ratio = df[BINARY_LABEL_COLUMN].mean()
        assert abs(train_ratio - overall_ratio) < 0.05
        assert abs(test_ratio - overall_ratio) < 0.05

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError):
            stratified_split(pd.DataFrame({"url": ["a"]}))

    def test_too_few_examples_per_class_raises(self):
        df = pd.DataFrame(
            {"url": ["a", "b", "c"], BINARY_LABEL_COLUMN: [0, 0, 1]}
        )
        with pytest.raises(ValueError):
            stratified_split(df)
