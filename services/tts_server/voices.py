"""
Synctona TTS-сервис — управление голосами (Step: клонирование).

Роутер для загрузки/удаления образцов голоса в пул. Образец кладётся в том
/voices/{voice_id}.wav, а его латенты кэшируются движком (precompute_speaker).
Дальше синтез по этому voice_id идёт уже склонированным голосом.

Подключается в app.py двумя строками (см. инструкцию), чтобы не раздувать app.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("synctona.tts.voices")


def make_voices_router(engine, voices_dir: str) -> APIRouter:
    os.makedirs(voices_dir, exist_ok=True)
    router = APIRouter()

    @router.post("/v1/voices/{voice_id}")
    async def upload_voice(voice_id: str, request: Request) -> JSONResponse:
        """Принимает сырые байты wav (application/octet-stream), кэширует голос."""
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="empty audio body")
        path = os.path.join(voices_dir, f"{_safe(voice_id)}.wav")
        with open(path, "wb") as f:
            f.write(data)
        try:
            await engine.precompute_speaker(voice_id, path)
        except Exception as e:  # noqa: BLE001
            logger.exception("precompute_speaker упал для %s", voice_id)
            raise HTTPException(status_code=500, detail=f"precompute failed: {e}")
        logger.info("Голос загружен и закэширован: %s", voice_id)
        return JSONResponse({"voice_id": voice_id, "ok": True})

    @router.get("/v1/voices/{voice_id}")
    async def has_voice(voice_id: str) -> JSONResponse:
        path = os.path.join(voices_dir, f"{_safe(voice_id)}.wav")
        return JSONResponse({"voice_id": voice_id, "exists": os.path.exists(path)})

    @router.delete("/v1/voices/{voice_id}")
    async def delete_voice(voice_id: str) -> JSONResponse:
        path = os.path.join(voices_dir, f"{_safe(voice_id)}.wav")
        if os.path.exists(path):
            os.remove(path)
        engine.remove_speaker(voice_id)
        logger.info("Голос удалён: %s", voice_id)
        return JSONResponse({"voice_id": voice_id, "ok": True})

    return router


async def load_existing_voices(engine, voices_dir: str) -> None:
    """При старте подхватывает уже загруженные голоса из тома (переживают рестарт)."""
    if not os.path.isdir(voices_dir):
        return
    for fn in os.listdir(voices_dir):
        if fn.endswith(".wav"):
            voice_id = fn[:-4]
            try:
                await engine.precompute_speaker(voice_id, os.path.join(voices_dir, fn))
                logger.info("Восстановлен голос из тома: %s", voice_id)
            except Exception:
                logger.exception("Не удалось восстановить голос %s", voice_id)


def _safe(voice_id: str) -> str:
    """Защита от path traversal в имени файла."""
    return "".join(c for c in voice_id if c.isalnum() or c in ("_", "-"))[:128] or "default"