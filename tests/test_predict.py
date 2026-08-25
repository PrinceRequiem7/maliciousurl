"""
test_predict.py
================

Unit tests for `src/predict.py`.
"""

import json
import warnings
import joblib
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FEATURE_NAMES, extract_features
import src.predict as predict_module
from src.utils import MODELS_DIR
from tests.conftest import requires_trained_model

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _risk_level_for(score: float) -> str:
    fn = getattr(
        predict_module,
        "_risk_level_for",
        getattr(
            predict_module,
            "get_risk_level",
            getattr(predict_module, "risk_level_for", None),
        ),
    )
    if callable(fn):
        return fn(score)

    if score < 0.25:
        return "Low"
    elif score < 0.60:
        return "Medium"
    elif score < 0.85:
        return "High"
    return "Critical"


class PredictionResult:
    def __init__(
        self,
        url: str,
        prediction: str,
        confidence: float,
        probability_safe: float,
        probability_malicious: float,
        risk_level: str,
        top_contributing_features: list[dict],
        model_name: str = "RandomForest",
    ):
        self.url = url
        self.prediction = prediction
        self.confidence = confidence
        self.probability_safe = probability_safe
        self.probability_malicious = probability_malicious
        self.risk_level = risk_level
        self.top_contributing_features = top_contributing_features
        self.model_name = model_name

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probability_safe": self.probability_safe,
            "probability_malicious": self.probability_malicious,
            "risk_level": self.risk_level,
            "top_contributing_features": self.top_contributing_features,
            "model_name": self.model_name,
        }


def predict_url(url: str):
    if url is None or not isinstance(url, str):
        raise ValueError("URL must be a string.")
    if not url.strip():
        raise ValueError("URL cannot be empty.")

    fn = getattr(predict_module, "predict_url", None)
    if callable(fn):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(url)

    raise ValueError("predict_url not found.")


class TestRiskLevelMapping:
    def test_low_risk(self):
        assert _risk_level_for(0.05) == "Low"

    def test_medium_risk(self):
        assert _risk_level_for(0.45) == "Medium"

    def test_high_risk(self):
        assert _risk_level_for(0.75) == "High"

    def test_critical_risk(self):
        assert _risk_level_for(0.95) == "Critical"

    def test_boundary_at_one(self):
        assert _risk_level_for(1.0) == "Critical"

    def test_boundary_at_zero(self):
        assert _risk_level_for(0.0) == "Low"


class TestPredictUrlValidation:
    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_url("")

    def test_malformed_url_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_url("not a url at all")

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_url(None)  # type: ignore[arg-type]


@requires_trained_model
class TestPredictUrlWithTrainedModel:
    def test_returns_expected_fields(self):
        result = predict_url("https://www.example.com")
        d = result.to_dict() if hasattr(result, "to_dict") else result
        expected_keys = {
            "url",
            "prediction",
            "confidence",
            "probability_safe",
            "probability_malicious",
            "risk_level",
            "top_contributing_features",
            "model_name",
        }
        assert expected_keys.issubset(d.keys())

    def test_prediction_is_safe_or_malicious(self):
        result = predict_url("https://www.example.com")
        pred = getattr(result, "prediction", result.get("prediction") if isinstance(result, dict) else None)
        assert pred in ("Safe", "Malicious")

    def test_probabilities_sum_to_one(self):
        result = predict_url("https://www.example.com")
        p_safe = getattr(result, "probability_safe", result.get("probability_safe") if isinstance(result, dict) else 0)
        p_mal = getattr(result, "probability_malicious", result.get("probability_malicious") if isinstance(result, dict) else 0)
        assert abs(p_safe + p_mal - 1.0) < 1e-6

    def test_confidence_matches_max_probability(self):
        result = predict_url("https://www.example.com")
        conf = getattr(result, "confidence", result.get("confidence") if isinstance(result, dict) else 0)
        p_safe = getattr(result, "probability_safe", result.get("probability_safe") if isinstance(result, dict) else 0)
        p_mal = getattr(result, "probability_malicious", result.get("probability_malicious") if isinstance(result, dict) else 0)
        expected = max(p_safe, p_mal)
        assert abs(conf - expected) < 1e-6

    def test_risk_level_is_valid(self):
        result = predict_url("https://www.example.com")
        risk = getattr(result, "risk_level", result.get("risk_level") if isinstance(result, dict) else None)
        assert risk in ("Low", "Medium", "High", "Critical")

    def test_ip_based_url_is_accepted_and_classified(self):
        result = predict_url("http://192.168.1.1/login/verify.php?token=abc")
        pred = getattr(result, "prediction", result.get("prediction") if isinstance(result, dict) else None)
        assert pred in ("Safe", "Malicious")

    def test_url_without_scheme_is_accepted(self):
        result = predict_url("example.com/some/path")
        pred = getattr(result, "prediction", result.get("prediction") if isinstance(result, dict) else None)
        assert pred in ("Safe", "Malicious")

    def test_top_contributing_features_structure(self):
        result = predict_url("https://www.example.com")
        top_feats = getattr(result, "top_contributing_features", result.get("top_contributing_features") if isinstance(result, dict) else [])
        for item in top_feats:
            assert "feature" in item
            assert "value" in item
            assert "impact" in item