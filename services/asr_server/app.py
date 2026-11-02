"""
Synctona ASR-сервис — WS сервер (Step 3).

WS /v1/stream:
  → init JSON: {session_id, language, sample_rate}
  → бинарные PCM16-моно @16k фреймы
  ← события JSON:
      {"type":"partial","text":...}
      {"type":"final","text":...,"is_sentence_end":true}
      {"type":"endpoint"}

GET /healthz /readyz · POST /v1/warmup · GET /metrics
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from asr_server.engine import StreamingASREngine
from asr_server.events import TranscriptType, TranscriptEvent

logger = logging.getLogger("synctona.asr.app")
app = FastAPI(title="Synctona ASR", version="0.3.0")

_DEVICE = os.getenv("SYNCTONA_ASR_DEVICE", "cuda")
_MODEL = os.getenv("SYNCTONA_ASR_MODEL", "small")

_engine = StreamingASREngine(model_size=_MODEL, device=_DEVICE)
_ready = False

ASR_FINALIZE_MS = Histogram(
    "synctona_asr_finalize_ms", "Время от endpoint до финального текста (ms)",
    buckets=(50, 100, 150, 200, 300, 500, 800, 1200),
)
ASR_SESSIONS = Counter("synctona_asr_sessions_total", "Открытые ASR-сессии")

_TYPE_NAME = {
    TranscriptType.PARTIAL: "partial",
    TranscriptType.FINAL: "final",
    TranscriptType.ENDPOINT: "endpoint",
}


@app.on_event("startup")
async def _startup() -> None:
    global _ready
    try:
        await _engine.warmup()
        _ready = True
        logger.info("ASR готов.")
    except Exception:
        logger.exception("Прогрев ASR провалился.")


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


@app.get("/metrics")
async def metrics_endpoint() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/v1/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    if not _ready:
        await ws.send_json({"type": "error", "detail": "engine_not_ready"})
        await ws.close()
        return

    session = None
    try:
        init = await ws.receive_json()
        language = init.get("language", "ru")
        session = _engine.new_session(language)
        ASR_SESSIONS.inc()

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is None:
                # Управляющее сообщение (например, флаш).
                txt = message.get("text")
                if txt and '"finalize"' in txt:
                    for ev in await session.finalize():
                        await ws.send_json(_encode(ev))
                continue

            for ev in await session.push(data):
                await ws.send_json(_encode(ev))

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Ошибка ASR-стрима")
    finally:
        if session is not None:
            try:
                for ev in await session.finalize():
                    await ws.send_json(_encode(ev))
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass


def _encode(ev) -> dict:
    return {
        "type": _TYPE_NAME.get(ev.type, "partial"),
        "text": ev.text,
        "is_sentence_end": ev.is_sentence_end,
        "language": ev.language,
    }