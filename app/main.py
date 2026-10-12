"""Servicio REST educativo para inferencia con Random Forest."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, FiniteFloat

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "artifacts")))
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("clasificador_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = RotatingFileHandler(LOG_DIR / "servicio.log", maxBytes=500_000, backupCount=2, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
logger.propagate = False

model = None
metadata: dict = {}
metrics_lock = Lock()
monitor = {"requests_total": 0, "errors_total": 0, "latency_ms_total": 0.0, "by_status": Counter()}


class FeaturesIn(BaseModel):
    """30 mediciones numéricas en el orden indicado por GET /schema."""
    features: list[FiniteFloat] = Field(min_length=30, max_length=30)


class BatchIn(BaseModel):
    instances: list[FeaturesIn] = Field(min_length=1, max_length=100)


class PredictOut(BaseModel):
    class_id: Literal[0, 1]
    prediction: Literal["maligno", "benigno"]
    confidence: float
    probabilities: dict[str, float]
    model_version: str


@asynccontextmanager
async def lifespan(application: FastAPI):
    global model, metadata
    model_path = ARTIFACT_DIR / "model.joblib"
    metadata_path = ARTIFACT_DIR / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise RuntimeError("Modelo no encontrado. Ejecute: python scripts/train_model.py")
    # joblib/pickle: cargar únicamente modelos creados por nosotros y de origen confiable.
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logger.info("startup modelo=%s version=%s", metadata["model_name"], metadata["version"])
    yield
    logger.info("shutdown servicio")


app = FastAPI(
    title="API REST - Clasificación con Random Forest",
    description="Ejemplo didáctico FastAPI + scikit-learn. No utilizar para diagnóstico médico.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_and_count(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("excepcion_interna request_id=%s path=%s", request_id, request.url.path)
        response = JSONResponse(status_code=500, content={"detail": "Error interno", "request_id": request_id})
    elapsed = round((time.perf_counter() - start) * 1000, 3)
    with metrics_lock:
        monitor["requests_total"] += 1
        monitor["latency_ms_total"] += elapsed
        monitor["by_status"][str(response.status_code)] += 1
        if response.status_code >= 400:
            monitor["errors_total"] += 1
    logger.info("request_id=%s method=%s path=%s status=%s elapsed_ms=%.3f", request_id, request.method, request.url.path, response.status_code, elapsed)
    response.headers["X-Request-ID"] = request_id
    return response


def predict_one(payload: FeaturesIn) -> PredictOut:
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    X = np.asarray(payload.features, dtype=float).reshape(1, -1)
    predicted_class = int(model.predict(X)[0])
    predicted_probs = model.predict_proba(X)[0]
    index = {int(cls): i for i, cls in enumerate(model.classes_)}
    p_maligno = float(predicted_probs[index[0]])
    p_benigno = float(predicted_probs[index[1]])
    return PredictOut(
        class_id=predicted_class,
        prediction=metadata["target_labels"][str(predicted_class)],
        confidence=round(float(predicted_probs[index[predicted_class]]), 6),
        probabilities={"maligno": round(p_maligno, 6), "benigno": round(p_benigno, 6)},
        model_version=metadata["version"],
    )


@app.get("/", tags=["Información"])
def index():
    return {"message": "Servicio de clasificación activo", "docs": "/docs", "endpoints": ["/health", "/schema", "/example", "/predict", "/predict/batch", "/metrics"]}


@app.get("/health", tags=["Monitoreo"])
def health():
    return {"status": "ok" if model is not None else "unavailable", "model_loaded": model is not None, "model_version": metadata.get("version")}


@app.get("/schema", tags=["Información"])
def schema():
    return {"n_features": metadata["n_features"], "feature_names_in_order": metadata["feature_names"], "labels": metadata["target_labels"], "disclaimer": metadata["disclaimer"]}


@app.get("/example", tags=["Información"])
def example():
    return metadata["example"]


@app.post("/predict", response_model=PredictOut, tags=["Predicción"])
def predict(payload: FeaturesIn):
    return predict_one(payload)


@app.post("/predict/batch", response_model=list[PredictOut], tags=["Predicción"])
def predict_batch(payload: BatchIn):
    return [predict_one(item) for item in payload.instances]


@app.get("/metrics", tags=["Monitoreo"])
def metrics():
    with metrics_lock:
        total = monitor["requests_total"]
        errors = monitor["errors_total"]
        mean = monitor["latency_ms_total"] / total if total else 0.0
        by_status = dict(monitor["by_status"])
    return {"requests_total": total, "errors_total": errors, "error_rate": round(errors / total, 5) if total else 0.0, "avg_latency_ms": round(mean, 3), "by_status": by_status, "scope": "memory_per_process"}
