import os
import numpy as np
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

# ---------------- PATH SETUP ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
DATA_BT = os.path.join(ROOT, "data", "processed", "bt_sequences")
DATA_CTCR = os.path.join(ROOT, "data", "processed", "ctcr_sequences")
DATA_HEM = os.path.join(ROOT, "data", "processed", "rainfall")
DATA_CTP = os.path.join(ROOT, "data", "processed", "ctp")

XGB_PATH = os.path.join(MODEL_DIR, "xgb_cloudburst_real_tif.joblib")
CONV_PATH = os.path.join(MODEL_DIR, "convlstm_model.h5")


@st.cache_resource
def load_models():
    xgb_model = joblib.load(XGB_PATH)
    conv_model = load_model(CONV_PATH)
    return xgb_model, conv_model


xgb_model, conv_model = load_models()

# ---------------- BOUNDING BOX ----------------
LON_MIN = 76.98890439198583
LON_MAX = 80.98749376462197
LAT_MIN = 28.00994227650679
LAT_MAX = 31.018845225230788

# ---------------- UI ----------------
st.set_page_config(page_title="Cloudburst Prediction Dashboard", layout="wide")
st.title("Cloudburst Prediction Dashboard — Automatic Mode")
st.markdown("""
This mode automatically uses **all available timestamps** and **all lead hours (1–4 hr)**
to compute the cloudburst probability.
""")

if st.button("RUN PREDICTION"):
    st.info("Running model on all timestamps… please wait 10–20 seconds.")

    if not os.path.isdir(DATA_BT):
        st.error("BT data folder not found.")
        st.stop()

    timestamps = sorted([f.replace(".npy", "") for f in os.listdir(DATA_BT)])
    if not timestamps:
        st.error("No BT sequences found in processed directory.")
        st.stop()

    all_xgb_probs = []
    all_conv_probs = []

    for ts in timestamps:
        bt_path = os.path.join(DATA_BT, ts + ".npy")
        ctcr_path = os.path.join(DATA_CTCR, ts + ".npy")
        hem_path = os.path.join(DATA_HEM, ts + ".npy")
        ctp_path = os.path.join(DATA_CTP, ts + ".npy")

        if not all(os.path.exists(p) for p in [bt_path, ctcr_path, hem_path, ctp_path]):
            continue

        bt = np.load(bt_path)
        ctcr = np.load(ctcr_path)
        hem = np.load(hem_path)
        ctp = np.load(ctp_path)

        # 11 features for XGBoost
        features = np.array([
            np.mean(bt), np.std(bt), np.min(bt), np.max(bt),
            np.mean(ctcr), np.max(ctcr),
            np.mean(hem), np.max(hem),
            np.mean(ctp), np.min(ctp),
            float(np.max(hem)),  # rainfall_max
        ], dtype=np.float32).reshape(1, -1)

        xgb_prob = float(xgb_model.predict_proba(features)[0, 1])
        all_xgb_probs.append(xgb_prob)

        # ConvLSTM expects (1, 4, 128, 128, 1)
        bt_for_conv = bt.reshape(1, bt.shape[0], bt.shape[1], bt.shape[2], 1)
        conv_map = conv_model.predict(bt_for_conv, verbose=0)[0]
        conv_map_norm = (conv_map - np.min(conv_map)) / (np.ptp(conv_map) + 1e-6)
        all_conv_probs.append(float(np.max(conv_map_norm)))

    if not all_xgb_probs or not all_conv_probs:
        st.error("No predictions were generated.")
        st.stop()

    # Fusion
    final_prob = 0.6 * np.mean(all_xgb_probs) + 0.4 * np.mean(all_conv_probs)
    st.success(f"Final Cloudburst Probability: **{final_prob * 100:.2f}%**")
