# AEROSCOPE — Aircraft Engine Diagnostics

Streamlit dashboard and ML training pipeline for NASA's C-MAPSS turbofan degradation data. The dataset is not bundled; upload an official FD001–FD004 subset through the app.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Upload a `train_FD00*.txt` file to explore telemetry and train/evaluate models. Optionally upload the matching `test_FD00*.txt` and `RUL_FD00*.txt` files to diagnose test engines against their published RUL labels. Model artifacts are written to `models/`.

Train from the command line:

```powershell
python -m src.train data\train_FD001.txt
```

## Pipeline

- Parse and validate NASA whitespace-delimited data; calculate capped training RUL and join optional test RUL labels.
- Remove constant features based on the training-engine split and scale telemetry.
- Compare Linear Regression, Random Forest, and Gradient Boosting for RUL using MAE, RMSE, and R².
- Compare Logistic Regression, Decision Tree, Random Forest, and SVM for health risk using accuracy, precision, recall, F1, and confusion matrix.
- Split by engine (not by individual sensor row) to avoid putting cycles from the same engine into both evaluation sets.

Health bands are derived from the RUL target: Healthy > 60 cycles, Warning 26–60, Critical ≤ 25. This educational tool is not a maintenance release decision.

The existing `frontend/` React dashboard and `backend/` FastAPI API remain available as a separate earlier demo; their generated model artifact is synthetic and is not used by this C-MAPSS Streamlit workflow.
