"""
conftest.py
===========

Ensures the project root is on `sys.path` so `from src.xxx import yyy`
imports resolve correctly no matter where `pytest` is invoked from, and
defines shared fixtures used across the test suite.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import MODELS_DIR  # noqa: E402


def _model_artifacts_present() -> bool:
    required = ["best_model.pkl", "scaler.pkl", "feature_names.json", "best_model_metadata.json"]
    return all((MODELS_DIR / name).exists() for name in required)


requires_trained_model = pytest.mark.skipif(
    not _model_artifacts_present(),
    reason="Trained model artifacts not found in models/. Run `python -m src.train` first.",
)
