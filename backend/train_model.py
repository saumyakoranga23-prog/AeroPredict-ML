from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from services.gradient_model import GradientBoostingRegressor

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


rng = np.random.default_rng(42)

engine_count = 180
cycles_per_engine = 140
samples = []

for engine_id in range(1, engine_count + 1):
    engine_bias = engine_id * 1.7
    for cycle in range(1, cycles_per_engine + 1):
        base = 95.0 - (cycle * 0.5) + (engine_bias * 0.2)
        setting_1 = 100.0 + 0.87 * cycle + rng.normal(0, 1.2) + engine_id * 0.06
        setting_2 = 40.0 + 0.18 * cycle + rng.normal(0, 0.45) + engine_id * 0.03
        setting_3 = 20.0 + 0.10 * cycle + rng.normal(0, 0.3) + engine_id * 0.02

        sensor_values = []
        for index in range(1, 22):
            signal = base / (index + 2.5)
            degradation = cycle * (0.35 + index * 0.05)
            noise = rng.normal(0, 1.2)
            sensor_value = signal + degradation + noise + engine_bias
            sensor_values.append(float(sensor_value))

        row = {
            "cycle": float(cycle),
            "setting_1": float(setting_1),
            "setting_2": float(setting_2),
            "setting_3": float(setting_3),
            **{f"sensor_{index}": float(value) for index, value in enumerate(sensor_values, start=1)},
        }
        samples.append(row)

X = np.array([[row[column] for column in FEATURE_COLUMNS] for row in samples], dtype=float)
y = np.array([
    max(0.0, min(125.0, 125.0 - (sample["cycle"] * 0.75) + (idx * 0.12) + rng.normal(0, 2.5)))
    for idx, sample in enumerate(samples, start=1)
], dtype=float)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X, y)

output_dir = Path(__file__).resolve().parent / "model"
output_dir.mkdir(exist_ok=True)

joblib.dump(model, output_dir / "aircraft_rul_model.pkl")
joblib.dump(FEATURE_COLUMNS, output_dir / "feature_columns.pkl")
joblib.dump(125, output_dir / "rul_cap.pkl")

print(f"Saved model artifacts to {output_dir}")
print(f"Model feature count: {len(FEATURE_COLUMNS)}")
print(f"Sample prediction: {model.predict(X[:1])[0]:.2f}")
