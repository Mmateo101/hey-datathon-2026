import joblib
import pandas as pd
from pathlib import Path


def save_model(model, path: str, name: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    full_path = f"{path}/{name}.joblib"
    joblib.dump(model, full_path)
    return full_path


def load_model(path: str):
    return joblib.load(path)
