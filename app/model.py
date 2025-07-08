import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "sentiment_model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)
