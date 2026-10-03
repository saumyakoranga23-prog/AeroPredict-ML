from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.model_service import FEATURE_COLUMNS, generate_sample_engine, get_model_status, load_model_artifacts, predict_rul

app = FastAPI(title="AEROSCOPE RUL API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    MODEL, FEATURE_LIST, RUL_CAP = load_model_artifacts()
except FileNotFoundError:
    MODEL = None
    FEATURE_LIST = FEATURE_COLUMNS
    RUL_CAP = 125.0


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    return {
        "status": "online",
        "model_loaded": bool(MODEL is not None),
        "rul_cap": RUL_CAP,
    }


@app.get("/api/sample-engine")
def get_sample_engine(engine_id: int = 1) -> dict[str, Any]:
    engine_id = max(1, min(engine_id, 10))
    payload = generate_sample_engine(engine_id)
    return {
        "engine": payload,
        "features": {name: payload.get(name, 0.0) for name in FEATURE_LIST},
    }


@app.get("/api/model/feature-importance")
def get_feature_importance() -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is unavailable.")

    if not hasattr(MODEL, "feature_importances_"):
        return {
            "available": False,
            "message": "Available from training notebook",
            "items": [],
        }

    importance_pairs = list(zip(FEATURE_LIST, MODEL.feature_importances_))
    ordered = sorted(importance_pairs, key=lambda item: item[1], reverse=True)
    return {
        "available": True,
        "items": [
            {"feature": feature, "importance": round(float(score), 6)}
            for feature, score in ordered[:10]
        ],
    }


@app.post("/api/predict")
async def post_predict(payload: dict[str, Any]) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Prediction service is unavailable. The model artifacts could not be loaded.")

    try:
        response = predict_rul(payload, MODEL, FEATURE_LIST, RUL_CAP)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for unexpected API failures
        raise HTTPException(status_code=500, detail=f"Prediction unavailable: {str(exc)}") from exc


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AEROSCOPE backend is online."}
