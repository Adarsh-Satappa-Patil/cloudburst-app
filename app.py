
import os
import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Cloudburst AI Demo",
    layout="wide"
)

st.title("Cloudburst Prediction — AI/ML Demo")

st.warning(
    "Experimental demonstration only. This model uses synthetic training "
    "data and is not validated for real-world rainfall or cloudburst warnings."
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    ROOT, "models", "xgb_cloudburst_real_tif.joblib"
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found in the models folder.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Enter six model input features")
st.caption(
    "These are illustrative inputs. Use the original training feature "
    "definitions and units before interpreting predictions."
)

c1, c2 = st.columns(2)

with c1:
    f1 = st.number_input("Feature 1", value=10.0)
    f2 = st.number_input("Feature 2", value=20.0)
    f3 = st.number_input("Feature 3", value=30.0)

with c2:
    f4 = st.number_input("Feature 4", value=40.0)
    f5 = st.number_input("Feature 5", value=50.0)
    f6 = st.number_input("Feature 6", value=60.0)

features = np.array([[f1, f2, f3, f4, f5, f6]])

if st.button("PREDICT RAINFALL", type="primary"):
    try:
        prediction = float(model.predict(features)[0])
        st.metric("Model-predicted rainfall", f"{prediction:.2f}")
        st.caption(
            "Output units depend on the model's training target. "
            "This is a model demonstration, not an operational forecast."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.write("Model status: Loaded")
st.write("Input features supplied:", features.shape[1])
st.write("Live satellite data: Not connected")

