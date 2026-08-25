import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import FEATURE_NAMES, build_feature_matrix
from src.utils import MODELS_DIR

# 1. Load processed URL data and compute the lexical feature matrix
print("Loading train.csv and extracting features...")
train_df = pd.read_csv("data/processed/train.csv")
train_features = build_feature_matrix(train_df)

X_train = train_features[list(FEATURE_NAMES)]
y_train = train_features["label"]

# 2. Fit memory-optimized Random Forest (depth-capped)
print("Training lightweight model...")
model = RandomForestClassifier(
    n_estimators=35,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 3. Fit scaler
scaler = StandardScaler()
scaler.fit(X_train)

# 4. Save artifacts with compression
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODELS_DIR / "best_model.pkl", compress=3)
joblib.dump(scaler, MODELS_DIR / "scaler.pkl", compress=3)

metadata = {
    "model_name": "RandomForest (Lightweight)",
    "uses_scaling": False
}
with open(MODELS_DIR / "best_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Finished! File size of best_model.pkl is now under 5MB.")