from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import (
    FAILURE_LABELS,
    FEATURE_COLUMNS,
    attach_test_rul,
    features_and_target,
    read_cmapss,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="AEROSCOPE | Engine Diagnostics", page_icon="✈️", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #07111f; color: #e7eef8; }
    [data-testid="stMetric"] { background: #101e30; padding: 16px; border: 1px solid #263a50; border-radius: 12px; }
    div[data-testid="stSidebar"] { background: #0b1727; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✈ AEROSCOPE")
st.caption("AIRCRAFT ENGINE DIAGNOSTICS  /  NASA C-MAPSS TURBOFAN DEGRADATION")

with st.sidebar:
    st.header("Dataset")
    train_file = st.file_uploader("Training data (train_FD00*.txt)", type=["txt"])
    test_file = st.file_uploader("Test data (test_FD00*.txt)", type=["txt"])
    rul_file = st.file_uploader("Test RUL labels (RUL_FD00*.txt)", type=["txt"])
    st.caption("Upload matching official NASA C-MAPSS FD001–FD004 files. Processed locally.")
    page = st.radio("Workspace", ["Diagnostics", "Model evaluation", "Data explorer"])

if not train_file:
    st.info("Upload a NASA `train_FD00*.txt` file to explore data and train the diagnostic models.")
    st.markdown(
        "The NASA C-MAPSS files are not bundled with this project. Download the dataset from "
        "[NASA's prognostics data repository](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data) "
        "and select the FD001 train/test/RUL files here."
    )
    st.stop()

try:
    train_frame = read_cmapss(train_file.getvalue().decode("utf-8"), "train")
    test_frame = read_cmapss(test_file.getvalue().decode("utf-8"), "test") if test_file else None
    if test_frame is not None and rul_file:
        test_frame = attach_test_rul(test_frame, rul_file.getvalue().decode("utf-8"))
except (UnicodeDecodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
    st.error(f"Could not read the uploaded dataset: {exc}")
    st.stop()

if "training_result" not in st.session_state:
    st.session_state.training_result = None

if st.button("Train and compare models", type="primary", use_container_width=True):
    try:
        from src.train import save_models, train_models

        with st.spinner("Training models and evaluating held-out engines..."):
            st.session_state.training_result = train_models(train_frame)
            save_models(st.session_state.training_result, MODEL_DIR)
        st.success("Training complete. Best models saved in `models/`.")
    except ImportError as exc:
        st.error(f"Could not load the ML runtime: {exc}")
    except ValueError as exc:
        st.error(f"Training could not complete: {exc}")

result = st.session_state.training_result
if result is None and (MODEL_DIR / "rul_model.pkl").exists():
    try:
        result = {
            "rul_model": joblib.load(MODEL_DIR / "rul_model.pkl"),
            "failure_model": joblib.load(MODEL_DIR / "failure_model.pkl"),
            "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
            "feature_columns": joblib.load(MODEL_DIR / "feature_columns.pkl"),
            "rul_cap": joblib.load(MODEL_DIR / "rul_cap.pkl"),
            **(
                joblib.load(MODEL_DIR / "metrics.pkl")
                if (MODEL_DIR / "metrics.pkl").exists()
                else {}
            ),
        }
    except (OSError, ValueError, KeyError, ImportError) as exc:
        st.error(f"Could not load saved models: {exc}")

if page == "Data explorer":
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training engines", train_frame["engine_id"].nunique())
    c2.metric("Training observations", f"{len(train_frame):,}")
    c3.metric("Telemetry features", len(FEATURE_COLUMNS))
    st.write("Cycle distribution")
    st.bar_chart(train_frame.groupby("engine_id")["cycle"].max(), height=240)
    st.write("Sensor trends by cycle")
    sensors = st.multiselect("Sensors", FEATURE_COLUMNS[3:], default=["sensor_2", "sensor_7", "sensor_14"])
    if sensors:
        st.line_chart(train_frame.groupby("cycle")[sensors].mean(), height=320)
    st.dataframe(train_frame.head(20), use_container_width=True)

elif page == "Model evaluation":
    st.subheader("Model comparison")
    if result is None or "regression_results" not in result:
        st.warning("Train the models to see held-out engine results.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown(f"**RUL regression — selected: {result['best_regressor']}**")
            st.dataframe(pd.DataFrame(result["regression_results"]).T, use_container_width=True)
        with right:
            st.markdown(f"**Failure classification — selected: {result['best_classifier']}**")
            st.dataframe(pd.DataFrame(result["classification_results"]).T.drop(columns="Confusion Matrix"), use_container_width=True)
            matrix = result["classification_results"][result["best_classifier"]]["Confusion Matrix"]
            st.caption("Confusion matrix (actual rows × predicted columns): Healthy, Warning, Critical")
            st.dataframe(pd.DataFrame(matrix, index=FAILURE_LABELS, columns=FAILURE_LABELS), use_container_width=True)

else:
    st.subheader("Engine health assessment")
    source = test_frame if test_frame is not None else train_frame
    engine_ids = sorted(source["engine_id"].unique())
    engine_id = st.selectbox("Engine ID", engine_ids)
    engine_rows = source[source["engine_id"] == engine_id].sort_values("cycle")
    row = engine_rows.iloc[-1]
    a, b = st.columns([1, 2])
    a.metric("Engine ID", int(engine_id))
    a.metric("Current cycle", int(row["cycle"]))
    if result is None:
        a.warning("Train or load the models to run a diagnosis.")
    elif a.button("RUN ENGINE DIAGNOSTICS", type="primary", use_container_width=True):
        feature_order = result["feature_columns"]
        observation = row[feature_order].to_frame().T.astype(float)
        scaled = result["scaler"].transform(observation)
        predicted_rul = float(result["rul_model"].predict(scaled)[0])
        predicted_rul = min(float(result["rul_cap"]), max(0.0, predicted_rul))
        predicted_health = result["failure_model"].predict(scaled)[0]
        st.session_state.diagnosis = {
            "engine_id": int(engine_id),
            "cycle": int(row["cycle"]),
            "predicted_rul": predicted_rul,
            "health": predicted_health,
            "actual_rul": float(row["rul"]) if "rul" in row else None,
        }

    diagnosis = st.session_state.get("diagnosis")
    if diagnosis and diagnosis["engine_id"] == int(engine_id) and diagnosis["cycle"] == int(row["cycle"]):
        a.metric("Predicted RUL", f"{diagnosis['predicted_rul']:.1f} cycles")
        a.metric("Engine health", diagnosis["health"])
        if diagnosis["actual_rul"] is not None:
            a.caption(f"Ground-truth test RUL: {diagnosis['actual_rul']:.0f} cycles")
    with b:
        st.markdown("**Engine telemetry**")
        st.line_chart(engine_rows.set_index("cycle")[FEATURE_COLUMNS[3:]], height=320)
    with st.expander("Current sensor readings"):
        st.dataframe(row[FEATURE_COLUMNS].to_frame("Reading"), use_container_width=True)

st.divider()
st.caption("Risk bands: Healthy > 60 cycles · Warning 26–60 cycles · Critical ≤ 25 cycles. "
           "This educational diagnostic is not an aircraft maintenance release.")
