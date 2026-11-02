"""
Synctona — сервис enrollment (Step: клонирование).

Единственное место, где фиксируется СОГЛАСИЕ на клонирование (§3/§7 ТЗ) и
связывается identity → voice_id. Клиент (моб/веб) перед сессией:
  1) записывает ~10–15с голоса,
  2) шлёт сюда с флагом consent=true,
  3) получает voice_id, который агент потом достанет из Redis по consent.

Оркестрирует: приём образца → заливка в TTS-пул → запись consent+voice в Redis.
Ключи Redis ДОЛЖНЫ совпадать с synctona/clients/speaker_registry.py.
"""

from __future__ import annotations

import logging
import os

import aiohttp
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

logger = logging.getLogger("synctona.enrollment")
app = FastAPI(title="Synctona Enrollment", version="0.1.0")

TTS_URL = os.getenv("SYNCTONA_TTS_ENDPOINT", "http://tts:8001").rstrip("/")
REDIS_URL = os.getenv("SYNCTONA_REDIS_URL", "redis://redis:6379/0")
TTL_S = int(os.getenv("SYNCTONA_VOICE_TTL_S", "86400"))

# Должно совпадать со схемой ключей в speaker_registry.py.
_VOICE_KEY = "synctona:voice:{}"
_CONSENT_KEY = "synctona:consent:{}"

_redis = None


async def _rc():
    global _redis
    if _redis is None:
        import redis.asyncio as redis  # type: ignore

        _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


@app.post("/v1/enroll")
async def enroll(
    identity: str = Form(...),
    consent: bool = Form(...),
    audio: UploadFile = File(...),
) -> JSONResponse:
    if not consent:
        raise HTTPException(status_code=400,
                            detail="Клонирование запрещено без явного согласия (consent=true).")

    voice_id = f"voice_{_safe(identity)}"
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Пустой аудио-файл.")

    # 1) Залить образец в TTS-пул (там он кэшируется).
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{TTS_URL}/v1/voices/{voice_id}",
                data=data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=502,
                                        detail=f"TTS voice upload failed: {await resp.text()}")
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=502, detail=f"TTS недоступен: {e}")

    # 2) Записать consent + связку identity→voice_id в Redis.
    try:
        rc = await _rc()
        pipe = rc.pipeline()
        pipe.set(_VOICE_KEY.format(identity), voice_id, ex=TTL_S)
        pipe.set(_CONSENT_KEY.format(identity), "1", ex=TTL_S)
        await pipe.execute()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Redis запись не удалась: {e}")

    logger.info("Enrolled: %s → %s (consent=1)", identity, voice_id)
    return JSONResponse({"identity": identity, "voice_id": voice_id, "consent": True})


@app.delete("/v1/enroll/{identity}")
async def revoke(identity: str) -> JSONResponse:
    voice_id = f"voice_{_safe(identity)}"
    # Убрать голос из пула (best-effort).
    try:
        async with aiohttp.ClientSession() as s:
            await s.delete(f"{TTS_URL}/v1/voices/{voice_id}",
                           timeout=aiohttp.ClientTimeout(total=10))
    except Exception:
        logger.warning("Не удалось удалить голос из TTS для %s", identity)
    # Снять consent.
    rc = await _rc()
    pipe = rc.pipeline()
    pipe.delete(_VOICE_KEY.format(identity))
    pipe.set(_CONSENT_KEY.format(identity), "0", ex=TTL_S)
    await pipe.execute()
    logger.info("Revoked: %s", identity)
    return JSONResponse({"identity": identity, "ok": True})


def _safe(identity: str) -> str:
    return "".join(c for c in identity if c.isalnum() or c in ("_", "-"))[:128] or "anon"