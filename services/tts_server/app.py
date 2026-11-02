"""
Synctona TTS-сервис — HTTP/WS сервер (Step 2).

Эндпоинты:
  WS   /v1/synthesize          — стрим PCM16-чанков по одному запросу
  POST /v1/cancel/{job_id}     — кооперативная отмена (шаг 2 barge-in)
  GET  /v1/utilization         — {"utilization": float} для admission/деградации
  POST /v1/warmup              — прогрев движка (idempotent)
  GET  /healthz                — liveness
  GET  /readyz                 — readiness (движок прогрет)
  GET  /metrics                — Prometheus

Протокол WS:
  → init JSON: {job_id, text, voice_id, ref_wav_path?, preset, sample_rate}
  ← бинарные фреймы: PCM16 mono chunks
  ← финал JSON: {"event": "eof"}  |  {"event": "error", "detail": ...}
                {"event": "rejected", "detail": ...}  (backpressure)
  клиент может прислать {"event": "cancel"} в сокет (эквивалент REST cancel).
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from tts_server.engine import ChatterboxEngine, SynthRequest
from tts_server.scheduler import RejectedError, TTSScheduler
from tts_server.voices import make_voices_router, load_existing_voices
logger = logging.getLogger("synctona.tts.app")

app = FastAPI(title="Synctona TTS Pool", version="0.2.0")

# Конфиг сервиса через окружение (в проде — из общего config, здесь автономно).
_DEVICE = os.getenv("SYNCTONA_TTS_DEVICE", "cuda")
_MAX_STREAMS = int(os.getenv("SYNCTONA_TTS_MAX_STREAMS_PER_GPU", "16"))
_GPUS = int(os.getenv("SYNCTONA_TTS_SERVING_GPU_COUNT", "1"))
_QUEUE_MAX = int(os.getenv("SYNCTONA_TTS_QUEUE_MAX", "32"))
_MAX_AGE_MS = int(os.getenv("SYNCTONA_TTS_MAX_QUEUE_AGE_MS", "800"))

_engine = ChatterboxEngine(device=_DEVICE)
_scheduler = TTSScheduler(
    _engine,
    max_streams_per_gpu=_MAX_STREAMS,
    serving_gpu_count=_GPUS,
    queue_max=_QUEUE_MAX,
    max_queue_age_ms=_MAX_AGE_MS,
)
_ready = asyncio.Event()

VOICES_DIR = os.getenv("SYNCTONA_VOICES_DIR", "/voices")
app.include_router(make_voices_router(_engine, VOICES_DIR))

@app.on_event("startup")
async def _startup() -> None:
    # Прогрев в фоне, чтобы сервер отвечал на healthz сразу, а readyz — по готовности.
    async def _bg() -> None:
        try:
            await _engine.warmup()
            await load_existing_voices(_engine, VOICES_DIR)
            _ready.set()
            logger.info("TTS-движок готов.")
        except Exception:
            logger.exception("Прогрев TTS-движка провалился.")

    asyncio.create_task(_bg())


@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.get("/readyz")
async def readyz() -> JSONResponse:
    if _ready.is_set():
        return JSONResponse({"ready": True})
    return JSONResponse({"ready": False}, status_code=503)


@app.post("/v1/warmup")
async def warmup() -> JSONResponse:
    if not _ready.is_set():
        try:
            await _engine.warmup()
            _ready.set()
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "detail": str(e)}, status_code=500)
    return JSONResponse({"ok": True})


@app.get("/v1/utilization")
async def utilization() -> JSONResponse:
    return JSONResponse(
        {
            "utilization": _scheduler.utilization(),
            "active": _scheduler.active_count(),
            "capacity": _scheduler.capacity,
        }
    )


@app.post("/v1/cancel/{job_id}")
async def cancel(job_id: str) -> JSONResponse:
    ok = await _scheduler.cancel(job_id)
    return JSONResponse({"cancelled": ok})


@app.get("/metrics")
async def metrics_endpoint() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/v1/synthesize")
async def synthesize_ws(ws: WebSocket) -> None:
    await ws.accept()
    job_id = "?"
    cancel_forwarder: asyncio.Task | None = None
    try:
        init = await ws.receive_json()
        req = SynthRequest(
            job_id=init["job_id"],
            text=init["text"],
            voice_id=init.get("voice_id", "default"),
            ref_wav_path=init.get("ref_wav_path"),
            preset=init.get("preset", "default"),
            sample_rate=int(init.get("sample_rate", _engine.sample_rate)),
        )
        job_id = req.job_id

        if not _ready.is_set():
            await ws.send_json({"event": "error", "detail": "engine_not_ready"})
            return

        # Прогреваем кэш голоса (idempotent).
        await _engine.precompute_speaker(req.voice_id, req.ref_wav_path)

        # Параллельно слушаем сокет на сообщение cancel, не блокируя стрим.
        async def _listen_cancel() -> None:
            try:
                while True:
                    msg = await ws.receive_json()
                    if msg.get("event") == "cancel":
                        await _scheduler.cancel(job_id)
                        return
            except Exception:
                return

        cancel_forwarder = asyncio.create_task(_listen_cancel())

        try:
            async for chunk in _scheduler.synthesize(req):
                if chunk:
                    await ws.send_bytes(chunk)
            await ws.send_json({"event": "eof"})
        except RejectedError as e:
            await ws.send_json({"event": "rejected", "detail": str(e)})

    except WebSocketDisconnect:
        # Клиент отвалился → отменяем джобу, освобождаем GPU.
        await _scheduler.cancel(job_id)
        logger.info("WS-дисконнект, джоба %s отменена", job_id)
    except Exception:
        logger.exception("Ошибка в /v1/synthesize (job=%s)", job_id)
        with _silent():
            await ws.send_json({"event": "error", "detail": "internal"})
    finally:
        if cancel_forwarder is not None:
            cancel_forwarder.cancel()
        with _silent():
            await ws.close()


class _silent:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return True