import os
import joblib
import numpy as np
import xgboost as xgb

MODEL_PATH = os.path.join(os.getcwd(), "models", "xgb_model.pkl")


def load_xgb_model(path=None):
    """Load XGBoost model (Scikit wrapper or Booster)."""
    model_path = path if path else MODEL_PATH
    if not os.path.exists(model_path):
        print(f"XGBoost model not found: {model_path}")
        return None

    try:
        return joblib.load(model_path)
    except Exception:
        pass

    try:
        booster = xgb.Booster({"nthread": 1})
        booster.load_model(model_path)
        return booster
    except Exception:
        print("Failed to load XGBoost model")
        return None


def predict_xgb_proba(model, X):
    """Return probability for each row."""
    if model is None or X.size == 0:
        return np.zeros((X.shape[0],), dtype="float32")

    try:
        prob = model.predict_proba(X)[:, 1]
        return np.clip(prob, 0, 1).astype("float32")
    except Exception:
        pass

    try:
        dmat = xgb.DMatrix(X)
        prob = model.predict(dmat)
        return np.clip(prob, 0, 1).astype("float32")
    except Exception:
        print("XGBoost predict failed -> returning zeros")
        return np.zeros((X.shape[0],), dtype="float32")