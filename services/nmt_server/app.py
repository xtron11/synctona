"""
Synctona NMT-сервис — HTTP сервер (Step 3).

Эндпоинты:
  POST /v1/translate   — {text, src, tgt, beam?} → {translation, latency_ms}
  GET  /healthz /readyz
  POST /v1/warmup
  GET  /metrics        — Prometheus

Батчинг — внутри движка (много одновременных POST от разных сессий сливаются в
батч). Клиент (HttpNMTClient) шлёт по одному предложению — так контракт из Step 1.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from nmt_server.engine import M2MEngine

logger = logging.getLogger("synctona.nmt.app")

app = FastAPI(title="Synctona NMT", version="0.3.0")

_MODEL = os.getenv("SYNCTONA_NMT_MODEL", "facebook/m2m100_418M")
_DEVICE = os.getenv("SYNCTONA_NMT_DEVICE", "cuda")

_engine = M2MEngine(model_name=_MODEL, device=_DEVICE)
_ready = False

NMT_LATENCY_MS = Histogram(
    "synctona_nmt_latency_ms", "NMT latency per sentence (ms)",
    buckets=(10, 20, 40, 60, 80, 120, 200, 400, 800, 1500),
)
NMT_ERRORS = Counter("synctona_nmt_errors_total", "Ошибки перевода")


class TranslateIn(BaseModel):
    text: str
    src: str = "ru"
    tgt: str = "en"
    beam: int = 4


class TranslateOut(BaseModel):
    translation: str
    latency_ms: float


@app.on_event("startup")
async def _startup() -> None:
    global _ready
    try:
        await _engine.warmup()
        _ready = True
        logger.info("NMT готов.")
    except Exception:
        logger.exception("Прогрев NMT провалился.")


@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.get("/readyz")
async def readyz() -> JSONResponse:
    return JSONResponse({"ready": _ready}, status_code=200 if _ready else 503)


@app.post("/v1/warmup")
async def warmup() -> JSONResponse:
    global _ready
    try:
        await _engine.warmup()
        _ready = True
        return JSONResponse({"ok": True})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "detail": str(e)}, status_code=500)


@app.post("/v1/translate", response_model=TranslateOut)
async def translate(req: TranslateIn) -> JSONResponse:
    if not _ready:
        return JSONResponse({"detail": "not_ready"}, status_code=503)
    t0 = time.monotonic()
    try:
        out = await _engine.translate(req.text, req.src, req.tgt, beam_size=req.beam, context=[])
    except Exception as e:  # noqa: BLE001
        NMT_ERRORS.inc()
        logger.exception("Ошибка перевода")
        return JSONResponse({"detail": str(e)}, status_code=500)
    latency = (time.monotonic() - t0) * 1000
    NMT_LATENCY_MS.observe(latency)
    return JSONResponse({"translation": out, "latency_ms": latency})


@app.get("/metrics")
async def metrics_endpoint() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)