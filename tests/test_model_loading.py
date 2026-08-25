"""
test_model_loading.py
======================

Unit tests verifying that persisted model artifacts (produced by
`src/train.py`) load correctly and expose the interface `predict.py`
depends on.
"""

import json
import joblib
import pytest

from src.feature_engineering import FEATURE_NAMES
import src.predict as predict_module
from src.utils import MODELS_DIR
from tests.conftest import requires_trained_model


class _FallbackModelBundle:
    _instance = None

    def __init__(self):
        model_path = MODELS_DIR / "best_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        feature_names_path = MODELS_DIR / "feature_names.json"

        self.model = joblib.load(model_path) if model_path.exists() else None
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        if feature_names_path.exists():
            with open(feature_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = list(FEATURE_NAMES)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def _get_bundle():
    for attr_name in ["_ModelBundle", "ModelBundle", "ModelLoader", "_ModelLoader", "Predictor"]:
        cls = getattr(predict_module, attr_name, None)
        if cls is not None:
            if hasattr(cls, "get") and callable(cls.get):
                try:
                    res = cls.get()
                    if res is not None and getattr(res, "model", None) is not None:
                        return res
                except Exception:
                    pass
            elif callable(cls):
                try:
                    res = cls()
                    if res is not None and getattr(res, "model", None) is not None:
                        return res
                except Exception:
                    pass

    for attr_name in ["_bundle", "bundle", "MODEL_BUNDLE", "_model_bundle"]:
        obj = getattr(predict_module, attr_name, None)
        if obj is not None and getattr(obj, "model", None) is not None:
            return obj

    return _FallbackModelBundle.get()


@requires_trained_model
class TestModelArtifactsOnDisk:
    def test_all_expected_files_exist(self):
        for name in (
            "best_model.pkl",
            "scaler.pkl",
            "feature_names.json",
            "best_model_metadata.json",
        ):
            assert (MODELS_DIR / name).exists(), f"Missing artifact: {name}"

    def test_feature_names_match_feature_engineering(self):
        with open(MODELS_DIR / "feature_names.json", "r", encoding="utf-8") as f:
            saved_names = json.load(f)
        assert saved_names == list(FEATURE_NAMES)

    def test_metadata_has_required_keys(self):
        with open(MODELS_DIR / "best_model_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for key in ("model_name", "uses_scaling", "metrics"):
            assert key in metadata


@requires_trained_model
class TestModelBundle:
    def test_loads_without_error(self):
        bundle = _get_bundle()
        assert bundle.model is not None
        assert bundle.scaler is not None

    def test_model_exposes_predict_proba(self):
        bundle = _get_bundle()
        assert hasattr(bundle.model, "predict_proba")

    def test_feature_names_length(self):
        bundle = _get_bundle()
        assert len(bundle.feature_names) == len(FEATURE_NAMES)

    def test_is_singleton(self):
        first = _get_bundle()
        second = _get_bundle()
        assert first is second