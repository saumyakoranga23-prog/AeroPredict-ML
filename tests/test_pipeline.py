import unittest

import numpy as np

from src.preprocessing import attach_test_rul, failure_label, read_cmapss


def make_rows(engine_count=10, cycles=18):
    rows = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            telemetry = [float(engine_id + cycle * sensor * 0.1) for sensor in range(1, 22)]
            rows.append(" ".join(map(str, [engine_id, cycle, 0.1, 0.2, 0.3, *telemetry])))
    return "\n".join(rows)


class PipelineTest(unittest.TestCase):
    def test_parse_training_and_failure_bands(self):
        frame = read_cmapss(make_rows(engine_count=2, cycles=4), "train")
        self.assertEqual(frame["rul"].tolist(), [3, 2, 1, 0, 3, 2, 1, 0])
        self.assertEqual(
            [failure_label(rul) for rul in (125, 60, 25)],
            ["Healthy", "Warning", "Critical"],
        )

    def test_attach_final_test_rul_to_each_cycle(self):
        frame = read_cmapss(make_rows(engine_count=2, cycles=2), "test")
        labeled = attach_test_rul(frame, "5\n10\n")
        self.assertEqual(labeled["rul"].tolist(), [6, 5, 11, 10])

    def test_model_comparison_and_feature_selection(self):
        try:
            from src.train import train_models
        except ImportError as exc:
            if "Application Control policy" in str(exc):
                self.skipTest(f"scikit-learn native module blocked by Windows policy: {exc}")
            raise

        frame = read_cmapss(make_rows(cycles=70), "train")
        frame["sensor_21"] = 1.0
        result = train_models(frame)
        self.assertNotIn("sensor_21", result["feature_columns"])
        self.assertEqual(len(result["regression_results"]), 3)
        self.assertEqual(len(result["classification_results"]), 4)
        self.assertTrue(np.isfinite(result["regression_results"]["Linear Regression"]["MAE"]))
        observation = frame[result["feature_columns"]].iloc[:1]
        scaled = result["scaler"].transform(observation)
        self.assertGreaterEqual(result["rul_model"].predict(scaled)[0], 0)
        self.assertIn(result["failure_model"].predict(scaled)[0], ("Healthy", "Warning", "Critical"))


if __name__ == "__main__":
    unittest.main()
