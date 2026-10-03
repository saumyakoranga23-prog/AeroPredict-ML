from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import classification_metrics, regression_metrics
from src.preprocessing import FEATURE_COLUMNS, RUL_CAP, features_and_target, read_cmapss


def train_models(frame) -> dict[str, Any]:
    features, rul, failure = features_and_target(frame)
    groups = frame["engine_id"].to_numpy()
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(features, rul, groups)
    )
    x_train_all, x_test_all = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = rul.iloc[train_idx], rul.iloc[test_idx]
    f_train, f_test = failure.iloc[train_idx], failure.iloc[test_idx]

    selected_columns = x_train_all.columns[x_train_all.var() > 1e-12].tolist()
    if not selected_columns:
        raise ValueError("No varying telemetry features were found in the training data.")
    x_train, x_test = x_train_all[selected_columns], x_test_all[selected_columns]
    features = features[selected_columns]
    scaler = StandardScaler().fit(x_train)
    x_train_scaled, x_test_scaled = scaler.transform(x_train), scaler.transform(x_test)

    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42),
    }
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", min_samples_leaf=2, n_jobs=-1, random_state=42),
        "SVM": SVC(class_weight="balanced"),
    }

    regression_results = {}
    regression_models = {}
    for name, model in regressors.items():
        model.fit(x_train_scaled, y_train)
        prediction = np.clip(model.predict(x_test_scaled), 0, RUL_CAP)
        regression_results[name] = regression_metrics(y_test, prediction)
        regression_models[name] = model

    classification_results = {}
    classification_models = {}
    for name, model in classifiers.items():
        model.fit(x_train_scaled, f_train)
        classification_results[name] = classification_metrics(f_test, model.predict(x_test_scaled))
        classification_models[name] = model

    best_regressor_name = min(regression_results, key=lambda name: regression_results[name]["MAE"])
    best_classifier_name = max(classification_results, key=lambda name: classification_results[name]["F1-score"])

    # Refit the selected estimators and scaler on all available training engines.
    final_scaler = StandardScaler().fit(features)
    all_scaled = final_scaler.transform(features)
    final_regressor = regressors[best_regressor_name].__class__(**regressors[best_regressor_name].get_params())
    final_classifier = classifiers[best_classifier_name].__class__(**classifiers[best_classifier_name].get_params())
    final_regressor.fit(all_scaled, rul)
    final_classifier.fit(all_scaled, failure)

    return {
        "scaler": final_scaler,
        "rul_model": final_regressor,
        "failure_model": final_classifier,
        "feature_columns": selected_columns,
        "rul_cap": RUL_CAP,
        "best_regressor": best_regressor_name,
        "best_classifier": best_classifier_name,
        "regression_results": regression_results,
        "classification_results": classification_results,
    }


def save_models(result: dict[str, Any], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["rul_model"], model_dir / "rul_model.pkl")
    joblib.dump(result["failure_model"], model_dir / "failure_model.pkl")
    joblib.dump(result["scaler"], model_dir / "scaler.pkl")
    joblib.dump(result["feature_columns"], model_dir / "feature_columns.pkl")
    joblib.dump(result["rul_cap"], model_dir / "rul_cap.pkl")
    joblib.dump(
        {
            key: result[key]
            for key in (
                "best_regressor",
                "best_classifier",
                "regression_results",
                "classification_results",
            )
        },
        model_dir / "metrics.pkl",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C-MAPSS RUL and failure-risk models.")
    parser.add_argument("train_file", type=Path, help="Path to train_FD001.txt")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    frame = read_cmapss(args.train_file.read_text(encoding="utf-8"), dataset="train")
    result = train_models(frame)
    save_models(result, args.models_dir)
    print(f"Best RUL model: {result['best_regressor']}")
    print(f"Best classifier: {result['best_classifier']}")
    print(f"Saved trained models to {args.models_dir.resolve()}")


if __name__ == "__main__":
    main()
