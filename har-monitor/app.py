"""
HAR Monitor — FastAPI backend.

Serves the dashboard (public/) and a small prediction API. The Keras model was
exported to NumPy weights (scripts/export_model.py), so the deployed function
needs only FastAPI + NumPy and stays far below Vercel's bundle limit.

Run locally:   uvicorn app:app --reload      → http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import logging
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger("har-monitor")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
PUBLIC_DIR = ROOT / "public"

N_FEATURES = 561
MAX_BATCH_ROWS = 1000          # keeps each request well under Vercel's 4.5 MB body limit
LOW_CONFIDENCE = 0.70          # below this, a prediction is flagged for review


# --------------------------------------------------------------------------- #
# Model (loaded once per function instance)
# --------------------------------------------------------------------------- #
class HARModel:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        n = int(data["n_layers"])
        self.layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n)]
        self.classes: List[str] = [str(c) for c in data["classes"]]
        self.features: List[str] = [str(f) for f in data["features"]]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)          # ReLU (BatchNorm is folded into w, b)
        h = h - h.max(axis=1, keepdims=True)
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)


class ModelUnavailable(Exception):
    """A model file is missing or unreadable. Returned to clients as a 503."""


def _load(what: str, fn):
    try:
        return fn()
    except Exception as e:  # missing file, corrupt archive, bad JSON, …
        log.exception("could not load %s", what)
        raise ModelUnavailable(f"{what} could not be loaded: {type(e).__name__}") from e


# lru_cache does not cache exceptions, so a failed load is retried on the next request.
@lru_cache(maxsize=1)
def get_model() -> HARModel:
    return _load("model weights", lambda: HARModel(MODEL_DIR / "har_mlp.npz"))


@lru_cache(maxsize=1)
def get_metrics() -> dict:
    return _load("metrics", lambda: json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def get_samples():
    def read():
        with np.load(MODEL_DIR / "samples.npz", allow_pickle=False) as d:
            return d["X"].astype(np.float32), d["y"], d["subject"]
    return _load("demo samples", read)


def describe(probs: np.ndarray) -> dict:
    model = get_model()
    i = int(probs.argmax())
    return {
        "activity": model.classes[i],
        "confidence": round(float(probs[i]), 4),
        "low_confidence": bool(probs[i] < LOW_CONFIDENCE),
        "probabilities": {c: round(float(p), 4) for c, p in zip(model.classes, probs)},
    }


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
def _check_row(row: List[float]) -> List[float]:
    if len(row) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {len(row)}")
    if not all(math.isfinite(v) for v in row):
        raise ValueError("features must be finite numbers")
    return row


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="561 pre-computed UCI HAR features")

    @field_validator("features")
    @classmethod
    def validate(cls, v):
        return _check_row(v)


class BatchRequest(BaseModel):
    rows: List[List[float]] = Field(..., description="up to 1000 rows of 561 features")

    @field_validator("rows")
    @classmethod
    def validate(cls, v):
        if not v:
            raise ValueError("rows must not be empty")
        if len(v) > MAX_BATCH_ROWS:
            raise ValueError(f"at most {MAX_BATCH_ROWS} rows per request; split larger files")
        for r in v:
            _check_row(r)
        return v


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="HAR Monitor API",
    description="Human activity recognition (UCI HAR, 6 classes) for factory automation.",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


@app.exception_handler(ModelUnavailable)
async def model_unavailable(_: Request, exc: ModelUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    # Always answer with JSON instead of letting the function crash.
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}"})


@app.get("/api/health")
def health():
    try:
        m = get_model()
    except ModelUnavailable as e:
        return JSONResponse(status_code=503, content={"status": "degraded", "detail": str(e)})
    return {"status": "ok", "model": "keras-dense-v1", "classes": m.classes, "features": len(m.features)}


@app.get("/api/metrics")
def metrics():
    return get_metrics()


@app.get("/api/features")
def features():
    return {"features": get_model().features}


@app.post("/api/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    probs = get_model().predict_proba(np.array([req.features]))[0]
    out = describe(probs)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    return out


@app.post("/api/predict/batch")
def predict_batch(req: BatchRequest):
    t0 = time.perf_counter()
    model = get_model()
    probs = model.predict_proba(np.asarray(req.rows, dtype=np.float32))
    idx = probs.argmax(1)
    return {
        "count": len(req.rows),
        "predictions": [model.classes[i] for i in idx],
        "confidence": [round(float(p), 4) for p in probs.max(1)],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


@app.get("/api/stream")
def stream(n: int = Query(1, ge=1, le=60), seed: Optional[int] = None):
    """Classify real held-out test windows, as a stand-in for a live sensor feed."""
    X, y, subj = get_samples()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=n)
    t0 = time.perf_counter()
    probs = get_model().predict_proba(X[idx])
    latency = (time.perf_counter() - t0) * 1000 / n
    classes = get_model().classes
    items = []
    for k, i in enumerate(idx):
        d = describe(probs[k])
        d.update({"true_activity": classes[int(y[i])], "subject": int(subj[i]),
                  "correct": d["activity"] == classes[int(y[i])]})
        items.append(d)
    return {"windows": items, "latency_ms": round(latency, 3)}


@app.get("/api/{path:path}", include_in_schema=False)
def api_not_found(path: str):
    raise HTTPException(status_code=404, detail=f"Unknown API route: /api/{path}")


# Dashboard (mounted last so /api routes win). On Vercel the public/ folder is
# moved to the CDN at build time and is NOT inside the function, so only mount
# it when it exists; otherwise StaticFiles raises at import and every route 500s.
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
