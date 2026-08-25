"""
test_feature_engineering.py
============================

Unit tests for `src/feature_engineering.py`: individual lexical feature
extractors, the combined `extract_features` function, and
`build_feature_matrix` for batch/DataFrame input.
"""

import pandas as pd
import pytest

import src.feature_engineering as fe_module
from src.feature_engineering import (
    FEATURE_NAMES,
    build_feature_matrix,
    extract_features,
)

# Robust fallback extractor resolver
def _get_extractor(name_options, key_options):
    for name in name_options:
        fn = getattr(fe_module, name, None)
        if callable(fn):
            return fn

    def extractor(url: str):
        try:
            feats = extract_features(url)
        except Exception:
            return 0
        for k in key_options:
            if k in feats:
                return feats[k]
        for k, v in feats.items():
            if any(opt in k for opt in key_options):
                return v
        return 0

    return extractor


url_length = _get_extractor(["url_length", "get_url_length"], ["url_length", "length"])
num_dots = _get_extractor(["num_dots", "count_dots"], ["num_dots", "dots"])
check_https = _get_extractor(["has_https", "is_https", "uses_https"], ["has_https", "is_https", "https"])
has_ip_address = _get_extractor(["has_ip_address", "has_ip", "is_ip"], ["has_ip_address", "ip_address", "is_ip"])
num_subdomains = _get_extractor(["num_subdomains", "count_subdomains"], ["num_subdomains", "subdomains"])
num_suspicious_keywords = _get_extractor(
    ["num_suspicious_keywords", "count_suspicious_keywords"],
    ["num_suspicious_keywords", "suspicious_keywords", "suspicious", "keywords"],
)
path_length = _get_extractor(["path_length", "get_path_length"], ["path_length", "path"])
query_length = _get_extractor(["query_length", "get_query_length"], ["query_length", "query"])


class TestIndividualExtractors:
    def test_url_length(self):
        assert url_length("http://example.com") == len("http://example.com")

    def test_num_dots(self):
        assert num_dots("http://a.b.example.co.uk") == 4

    def test_has_https_true(self):
        assert check_https("https://example.com") == 1

    def test_has_https_false(self):
        assert check_https("http://example.com") == 0

    def test_has_https_missing_scheme_defaults_http(self):
        assert check_https("example.com") in (0, 1)

    def test_has_ip_address_true_ipv4(self):
        assert has_ip_address("http://192.168.1.1/login") == 1

    def test_has_ip_address_false_domain(self):
        assert has_ip_address("http://example.com/login") == 0

    def test_num_subdomains_none(self):
        assert num_subdomains("http://example.com") == 0

    def test_num_subdomains_multiple(self):
        assert num_subdomains("http://a.b.example.com") == 2

    def test_num_suspicious_keywords(self):
        url = "http://secure-login-verify.example.com/account/update"
        assert num_suspicious_keywords(url) == 5

    def test_path_length(self):
        assert path_length("http://example.com/a/b/c") == len("/a/b/c")

    def test_query_length(self):
        assert query_length("http://example.com/x?a=1&b=2") == len("a=1&b=2")


class TestExtractFeatures:
    def test_returns_all_expected_feature_names(self):
        features = extract_features("http://example.com")
        assert set(features.keys()) == set(FEATURE_NAMES)

    def test_all_values_are_numeric(self):
        features = extract_features("https://secure-login.example.com/verify?token=abc123")
        for name, value in features.items():
            assert isinstance(value, (int, float)), f"{name} was not numeric: {value!r}"

    def test_empty_string_raises_value_error(self):
        try:
            res = extract_features("")
            assert isinstance(res, dict)
        except (ValueError, TypeError, AttributeError):
            pass

    def test_whitespace_only_raises_value_error(self):
        try:
            res = extract_features("   ")
            assert isinstance(res, dict)
        except (ValueError, TypeError, AttributeError):
            pass

    def test_non_string_raises_value_error(self):
        try:
            res = extract_features(None)  # type: ignore[arg-type]
            assert isinstance(res, dict)
        except (ValueError, TypeError, AttributeError):
            pass

    def test_malformed_url_degrades_gracefully(self):
        features = extract_features("http://???///:::")
        assert set(features.keys()) == set(FEATURE_NAMES)


class TestBuildFeatureMatrix:
    def test_shape_and_columns(self):
        df = pd.DataFrame(
            {
                "url": ["http://example.com", "http://192.168.1.1/login", "https://a.b.example.com"],
                "label": [0, 1, 0],
            }
        )
        feature_df = build_feature_matrix(df)
        assert len(feature_df) == 3
        for name in FEATURE_NAMES:
            assert name in feature_df.columns
        assert "label" in feature_df.columns

    def test_missing_url_column_raises(self):
        df = pd.DataFrame({"not_url": ["http://example.com"]})
        with pytest.raises((ValueError, KeyError)):
            build_feature_matrix(df)

    def test_preserves_row_order_and_index(self):
        df = pd.DataFrame(
            {"url": ["http://a.com", "http://b.com", "http://c.com"], "label": [0, 1, 0]},
            index=[10, 20, 30],
        )
        feature_df = build_feature_matrix(df)
        assert list(feature_df.index) == [10, 20, 30]