
import os
import numpy as np

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

MODEL_PATH = os.path.join(os.getcwd(), "models", "convlstm_model.h5")


def load_convlstm():
    """Load ConvLSTM model safely."""
    if load_model is None:
        print("TensorFlow not available -> ConvLSTM disabled")
        return None
    if not os.path.exists(MODEL_PATH):
        print(f"ConvLSTM model missing -> {MODEL_PATH}")
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        print("Failed to load ConvLSTM:", e)
        return None


def predict_next_frame(model, frames):
    """
    frames: (T, H, W)
    returns: (H, W) predicted BT frame
    """
    if model is None:
        print("ConvLSTM model not loaded")
        return None
    if frames.ndim != 3:
        print("ERROR: ConvLSTM expects frames with shape (T, H, W)")
        return None

    x = frames.astype("float32").reshape(
        1, frames.shape[0], frames.shape[1], frames.shape[2], 1
    )
    pred = model.predict(x)[0, :, :, 0]
    pred = np.clip(pred, 180, 330)  # Kelvin
    return pred.astype("float32")