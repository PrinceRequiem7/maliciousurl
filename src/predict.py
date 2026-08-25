"""
predict.py
==========

Inference pipeline for URL safety classification.
Exposes `predict_url`, `ModelBundle`, `_ModelBundle`, and `_risk_level_for`.
"""

from __future__ import annotations

import json
import os
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from src.allowlist import is_trusted_domain
from src.feature_engineering import FEATURE_NAMES, extract_features
from src.utils import MODELS_DIR, get_logger

logger = get_logger(__name__)


def is_valid_url(url: str) -> bool:
    """Validate whether a string has a valid URL structure."""
    if not isinstance(url, str):
        return False
    cleaned = url.strip()
    if not cleaned or " " in cleaned or "\t" in cleaned or "\n" in cleaned:
        return False

    try:
        norm = cleaned if cleaned.startswith(("http://", "https://")) else "http://" + cleaned
        parsed = urlparse(norm)
        if not parsed.netloc:
            return False
        host = parsed.netloc.split(":")[0].strip("[]")
        if not host:
            return False
        if "." in host or host == "localhost":
            return True
        return False
    except Exception:
        return False


def _risk_level_for(prob_malicious: float) -> str:
    """Map a malicious probability score [0.0, 1.0] to a human-readable risk category."""
    if prob_malicious < 0.25:
        return "Low"
    elif prob_malicious < 0.60:
        return "Medium"
    elif prob_malicious < 0.85:
        return "High"
    return "Critical"


# Public alias for risk mapping
get_risk_level = _risk_level_for


@dataclass
class PredictionResult:
    """Container for structured prediction outputs."""
    url: str
    prediction: str  # "Safe" or "Malicious"
    confidence: float
    probability_safe: float
    probability_malicious: float
    risk_level: str
    top_contributing_features: List[Dict[str, Any]]
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelBundle:
    """Singleton loader for persisted model artifacts with remote-fetch fallback."""
    _instance: ModelBundle | None = None

    def __init__(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "best_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        feature_names_path = MODELS_DIR / "feature_names.json"
        metadata_path = MODELS_DIR / "best_model_metadata.json"

        # 1. Download model weights if missing (e.g. deployed on Render ephemeral storage)
        model_url = os.getenv("MODEL_URL")
        if not model_path.exists():
            if model_url:
                logger.info("Downloading best_model.pkl from %s...", model_url)
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    logger.info("Successfully downloaded best_model.pkl.")
                except Exception as e:
                    logger.error("Failed to download model from %s: %s", model_url, e)
                    raise FileNotFoundError(
                        f"Failed to fetch model from MODEL_URL: {e}"
                    ) from e
            else:
                raise FileNotFoundError(
                    f"Model artifact not found at {model_path}. Set MODEL_URL env "
                    "var or run training locally first."
                )

        # 2. Download scaler if missing and URL is specified
        scaler_url = os.getenv("SCALER_URL")
        if not scaler_path.exists() and scaler_url:
            logger.info("Downloading scaler.pkl from %s...", scaler_url)
            try:
                urllib.request.urlretrieve(scaler_url, scaler_path)
                logger.info("Successfully downloaded scaler.pkl.")
            except Exception as e:
                logger.warning("Could not download scaler.pkl from %s: %s", scaler_url, e)

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        if feature_names_path.exists():
            with open(feature_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = list(FEATURE_NAMES)

        self.uses_scaling = False
        self.model_name = "RandomForest"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.model_name = meta.get("model_name", "RandomForest")
                if "uses_scaling" in meta:
                    self.uses_scaling = bool(meta["uses_scaling"])
                else:
                    logger.warning(
                        "'uses_scaling' not found in best_model_metadata.json; "
                        "defaulting to False. If '%s' requires scaled input, "
                        "re-train to regenerate metadata.",
                        self.model_name,
                    )
        else:
            logger.warning(
                "best_model_metadata.json not found; defaulting uses_scaling=False."
            )

    @classmethod
    def get(cls) -> ModelBundle:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Private alias for compatibility
_ModelBundle = ModelBundle


def predict_url(raw_url: str) -> PredictionResult:
    """
    Predict whether a raw URL is Safe or Malicious.

    Args:
        raw_url: The URL string to analyze.

    Returns:
        PredictionResult dataclass with full classification details.
    """
    if raw_url is None or not isinstance(raw_url, str):
        raise ValueError("URL must be a non-empty string.")

    cleaned_url = raw_url.strip()
    if not cleaned_url or not is_valid_url(cleaned_url):
        raise ValueError(f"Invalid or malformed URL: {raw_url!r}")

    # 0. Trusted-domain allowlist check
    if is_trusted_domain(cleaned_url):
        logger.info("'%s' matched trusted-domain allowlist; bypassing model.", cleaned_url)
        return PredictionResult(
            url=raw_url,
            prediction="Safe",
            confidence=1.0,
            probability_safe=1.0,
            probability_malicious=0.0,
            risk_level="Low",
            top_contributing_features=[{
                "feature": "trusted_domain_allowlist",
                "value": "matched",
                "impact": 0.0,
            }],
            model_name="Allowlist (trusted domain)",
        )

    # 1. Feature extraction
    feats_dict = extract_features(cleaned_url)

    # 2. Load trained model bundle
    bundle = ModelBundle.get()

    # 3. Align input features with expected model schema
    df_features = pd.DataFrame([feats_dict])[bundle.feature_names]

    # 4. Model Inference & Scaling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if bundle.uses_scaling and bundle.scaler is not None:
            scaled_array = bundle.scaler.transform(df_features)
            X = pd.DataFrame(scaled_array, columns=bundle.feature_names)
        else:
            X = df_features

        if hasattr(bundle.model, "predict_proba"):
            probs = bundle.model.predict_proba(X)[0]
            prob_safe = float(probs[0])
            prob_malicious = float(probs[1])
        else:
            pred = bundle.model.predict(X)[0]
            prob_malicious = float(pred)
            prob_safe = 1.0 - prob_malicious

    prediction = "Malicious" if prob_malicious >= 0.5 else "Safe"
    confidence = max(prob_safe, prob_malicious)
    risk_level = _risk_level_for(prob_malicious)

    # 5. Extract top feature contributions
    top_features: List[Dict[str, Any]] = []
    if hasattr(bundle.model, "feature_importances_"):
        importances = bundle.model.feature_importances_
        contributions = importances * X.iloc[0].to_numpy()
        sorted_indices = np.argsort(np.abs(contributions))[::-1][:5]
        for idx in sorted_indices:
            fname = bundle.feature_names[idx]
            top_features.append({
                "feature": fname,
                "value": feats_dict.get(fname, 0),
                "impact": float(contributions[idx]),
            })
    elif hasattr(bundle.model, "coef_"):
        coefs = np.ravel(bundle.model.coef_)
        contributions = coefs * X.iloc[0].to_numpy()
        sorted_indices = np.argsort(np.abs(contributions))[::-1][:5]
        for idx in sorted_indices:
            fname = bundle.feature_names[idx]
            top_features.append({
                "feature": fname,
                "value": feats_dict.get(fname, 0),
                "impact": float(contributions[idx]),
            })
    else:
        for fname in bundle.feature_names[:5]:
            top_features.append({
                "feature": fname,
                "value": feats_dict.get(fname, 0),
                "impact": abs(float(feats_dict.get(fname, 0))),
            })

    return PredictionResult(
        url=raw_url,
        prediction=prediction,
        confidence=confidence,
        probability_safe=prob_safe,
        probability_malicious=prob_malicious,
        risk_level=risk_level,
        top_contributing_features=top_features,
        model_name=bundle.model_name,
    )