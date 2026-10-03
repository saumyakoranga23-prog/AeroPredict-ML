from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd

RUL_CAP = 125
ID_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS + ["cycle"]
COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
FAILURE_LABELS = ["Healthy", "Warning", "Critical"]


def read_cmapss(text: str, dataset: str = "train") -> pd.DataFrame:
    """Parse a NASA C-MAPSS whitespace-delimited train or test file."""
    if dataset not in {"train", "test"}:
        raise ValueError("dataset must be 'train' or 'test'.")

    frame = pd.read_csv(StringIO(text), sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, found {frame.shape[1]}.")

    frame.columns = COLUMNS
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains missing or non-finite values.")
    if (frame[ID_COLUMNS] <= 0).any().any():
        raise ValueError("Engine IDs and cycles must be positive.")
    if frame.duplicated(ID_COLUMNS).any():
        raise ValueError("Dataset contains duplicate engine/cycle rows.")

    frame = frame.sort_values(ID_COLUMNS, ignore_index=True)
    if dataset == "train":
        last_cycle = frame.groupby("engine_id")["cycle"].transform("max")
        frame["rul"] = (last_cycle - frame["cycle"]).clip(upper=RUL_CAP)
    return frame


def attach_test_rul(test: pd.DataFrame, rul_text: str) -> pd.DataFrame:
    """Join NASA's per-engine test RUL labels to the final test rows."""
    try:
        label_frame = pd.read_csv(StringIO(rul_text), sep=r"\s+", header=None)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("RUL label file is empty.") from exc
    if label_frame.shape[1] != 1:
        raise ValueError("RUL label file must contain exactly one value per line.")
    labels = label_frame.iloc[:, 0]
    engine_ids = sorted(test["engine_id"].unique())
    if len(labels) != len(engine_ids):
        raise ValueError(f"Expected {len(engine_ids)} RUL labels, found {len(labels)}.")
    if not np.isfinite(labels.to_numpy(dtype=float)).all() or (labels < 0).any():
        raise ValueError("RUL labels must be finite and non-negative.")

    remaining = dict(zip(engine_ids, labels.astype(float)))
    result = test.copy()
    final_cycle = result.groupby("engine_id")["cycle"].transform("max")
    result["rul"] = [
        min(RUL_CAP, final_cycle.iloc[index] - row.cycle + remaining[row.engine_id])
        for index, row in enumerate(result.itertuples())
    ]
    return result


def failure_label(rul: float) -> str:
    if rul <= 25:
        return "Critical"
    if rul <= 60:
        return "Warning"
    return "Healthy"


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = frame[FEATURE_COLUMNS].astype(float)
    rul = frame["rul"].clip(upper=RUL_CAP).astype(float)
    return features, rul, rul.map(failure_label)
