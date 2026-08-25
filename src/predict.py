"""
predict.py
==========

Inference pipeline for URL safety classification.
Exposes `predict_url`, `ModelBundle`, `_ModelBundle`, and `_risk_level_for`.
"""

from __future__ import annotations

import json
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
    """Singleton loader for persisted model artifacts."""
    _instance: ModelBundle | None = None

    def __init__(self) -> None:
        model_path = MODELS_DIR / "best_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"
        feature_names_path = MODELS_DIR / "feature_names.json"
        metadata_path = MODELS_DIR / "best_model_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Run training first.")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        if feature_names_path.exists():
            with open(feature_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = list(FEATURE_NAMES)

        # --- FIX: read whether THIS SPECIFIC model actually needs scaled
        # input, instead of inferring it from whether scaler.pkl merely
        # exists on disk. train.py always writes scaler.pkl regardless of
        # which of the 5 candidate models wins (it's needed for Logistic
        # Regression / SVM but harmless to save even when the winner is a
        # tree-based model), so "self.scaler is not None" is ALWAYS true
        # and is not a valid signal for whether to apply it. The previous
        # version of this class didn't read uses_scaling at all, which
        # meant every prediction silently ran features through a
        # StandardScaler even when the winning model (e.g. Random Forest)
        # was trained on raw, unscaled features -- corrupting every single
        # prediction without any error or warning.
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
                        "defaulting to False (no scaling applied). If '%s' actually "
                        "requires scaled input, predictions will be wrong. Re-run "
                        "training to regenerate metadata with this field.",
                        self.model_name,
                    )
        else:
            logger.warning(
                "best_model_metadata.json not found; defaulting uses_scaling=False. "
                "If the loaded model actually requires scaled input, predictions "
                "will be wrong."
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

    # 0. Trusted-domain allowlist check -- see src/allowlist.py for the full
    # rationale. Runs BEFORE any feature extraction or model inference, and
    # is fully transparent: the returned model_name makes it explicit that
    # this result bypassed the trained classifier entirely.
    if is_trusted_domain(cleaned_url):
        logger.info("'%s' matched the trusted-domain allowlist; bypassing model.", cleaned_url)
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
        # --- FIX: gate scaling on bundle.uses_scaling, not merely on
        # whether a scaler object happens to be loaded. See the comment
        # in ModelBundle.__init__ for why "scaler is not None" was wrong.
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
    # --- FIX: the previous version ranked and reported ONLY
    # bundle.model.feature_importances_ -- a GLOBAL, model-wide ranking
    # that is identical for every single prediction regardless of that
    # URL's actual feature values (this is why, in the earlier diagnostic
    # output, "impact" for num_dots was exactly +0.2427 for every URL
    # even though num_dots itself varied between them). Multiplying each
    # feature's global importance by that URL's OWN value turns this into
    # a real (if approximate) per-instance explanation: a feature the
    # model considers important AND that has an unusual value for this
    # specific URL will now correctly rank higher than one that's
    # "important in general" but unremarkable for this particular input.
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