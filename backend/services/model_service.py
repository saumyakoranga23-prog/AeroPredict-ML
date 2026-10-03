from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "aircraft_rul_model.pkl"
FEATURE_PATH = MODEL_DIR / "feature_columns.pkl"
RUL_CAP_PATH = MODEL_DIR / "rul_cap.pkl"

FEATURE_COLUMNS = [
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


def load_model_artifacts() -> tuple[Any, list[str], float]:
    if not MODEL_PATH.exists() or not FEATURE_PATH.exists() or not RUL_CAP_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing from backend/model/")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    rul_cap = float(joblib.load(RUL_CAP_PATH))
    return model, feature_columns, rul_cap


def get_model_status() -> bool:
    return MODEL_PATH.exists() and FEATURE_PATH.exists() and RUL_CAP_PATH.exists()


def determine_dashboard_status(rul_value: float) -> str:
    if rul_value >= 100:
        return "healthy"
    if rul_value >= 60:
        return "monitor"
    if rul_value >= 25:
        return "warning"
    return "critical"


def validate_payload(payload: dict[str, Any], feature_columns: list[str]) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing_fields = [field for field in feature_columns if field not in payload]
    if missing_fields:
        raise ValueError(f"Missing required feature(s): {', '.join(missing_fields)}")

    cleaned: dict[str, float] = {}
    for field in feature_columns:
        value = payload.get(field)
        if value is None or value == "":
            raise ValueError(f"Field '{field}' cannot be empty.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field '{field}' must be numeric.") from exc
        if not math.isfinite(number):
            raise ValueError(f"Field '{field}' must be a finite number.")
        cleaned[field] = number

    return cleaned


def build_feature_frame(payload: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    validated = validate_payload(payload, feature_columns)
    return pd.DataFrame([validated], columns=feature_columns)


def predict_rul(payload: dict[str, Any], model: Any, feature_columns: list[str], rul_cap: float) -> dict[str, Any]:
    frame = build_feature_frame(payload, feature_columns)
    prediction = float(model.predict(frame)[0])
    bounded = float(np.clip(prediction, 0.0, rul_cap))
    status = determine_dashboard_status(bounded)
    return {
        "predicted_rul": round(bounded, 2),
        "unit": "cycles",
        "status": status,
        "rul_cap": rul_cap,
    }


def generate_sample_engine(engine_id: int) -> dict[str, Any]:
    engine_seed = (engine_id * 37) % 97
    cycle = 78 + (engine_seed % 55)
    values = {
        "engine_id": engine_id,
        "cycle": cycle,
        "setting_1": round(100.0 + 0.6 * cycle + (engine_seed % 7) * 0.4, 3),
        "setting_2": round(40.0 + 0.15 * cycle + (engine_seed % 11) * 0.2, 3),
        "setting_3": round(20.0 + 0.09 * cycle + (engine_seed % 9) * 0.3, 3),
    }

    for index in range(1, 22):
        waveform = math.sin((engine_id * 1.7) + (index * 0.8) + cycle / 18)
        offset = index * 0.25
        drift = (cycle / 120) * (index / 4)
        reading = 50 + waveform * 18 + offset + drift + (engine_seed % 5)
        values[f"sensor_{index}"] = round(reading, 3)

    return values
