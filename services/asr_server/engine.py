"""
Synctona ASR-сервис — движок (Step 3).

Стриминговое распознавание: скользящий буфер + Silero VAD для семантического
endpoint'а. Выдаёт партиалы по ходу речи и финал по концу предложения — именно
это позволяет держать min_endpointing_delay низким (§0/§2 ТЗ).

Реализация в духе WhisperLive (faster-whisper по накопленному буферу). Настоящий
стриминговый ASR (Parakeet/Canary streaming, Kyutai STT) подключается тем же
интерфейсом ASRSession и обычно даёт меньший TTFP (time-to-first-partial).

Каждое WS-соединение получает свою ASRSession (VAD stateful на поток).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterator, Optional

from asr_server.events import TranscriptEvent, TranscriptType

logger = logging.getLogger("synctona.asr.engine")

_VAD_FRAME = 512          # Silero @16k
_SR = 16000
_PARTIAL_EVERY_MS = 400   # как часто пересчитываем партиал


class StreamingASREngine:
    def __init__(self, *, model_size: str = "small", device: str = "cuda",
                 endpoint_silence_ms: int = 500) -> None:
        self._model_size = model_size
        self._device = device
        self._endpoint_silence_frames = int(endpoint_silence_ms / 1000 * _SR / _VAD_FRAME)
        self._model = None
        self._vad = None
        self._loaded = asyncio.Event()
        self._model_lock = asyncio.Lock()

    async def warmup(self) -> None:
        if self._loaded.is_set():
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_blocking)
        self._loaded.set()
        logger.info("ASR-движок прогрет (%s на %s)", self._model_size, self._device)

    def _load_blocking(self) -> None:
        import torch
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_size, device=self._device,
            compute_type="float16" if self._device.startswith("cuda") else "int8",
        )
        self._vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

    def new_session(self, language: str) -> "ASRSession":
        if not self._loaded.is_set():
            raise RuntimeError("ASR-движок не прогрет.")
        return ASRSession(self, language)


class ASRSession:
    """Состояние одного аудио-потока: буфер речи, VAD, эндпоинтинг."""

    def __init__(self, engine: StreamingASREngine, language: str) -> None:
        self._engine = engine
        self._lang = language
        self._speech: list[float] = []
        self._silence_frames = 0
        self._last_partial_ms = 0.0
        self._speaking = False

    async def push(self, pcm16: bytes) -> list[TranscriptEvent]:
        """Принимает PCM16-моно @16k, возвращает события (partial/final/endpoint)."""
        import numpy as np

        events: list[TranscriptEvent] = []
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        # Режем на VAD-кадры по 512.
        for i in range(0, len(audio) - _VAD_FRAME + 1, _VAD_FRAME):
            frame = audio[i:i + _VAD_FRAME]
            is_speech = await self._vad_is_speech(frame)
            if is_speech:
                self._speaking = True
                self._silence_frames = 0
                self._speech.extend(frame.tolist())
            elif self._speaking:
                self._speech.extend(frame.tolist())
                self._silence_frames += 1

        if not self._speaking:
            return events

        now_ms = time.monotonic() * 1000

        # Эндпоинт: достаточно тишины → финал предложения.
        if self._silence_frames >= self._engine._endpoint_silence_frames and self._speech:
            text = await self._transcribe(final=True)
            if text:
                events.append(TranscriptEvent(TranscriptType.FINAL, text, self._lang, True))
            events.append(TranscriptEvent(TranscriptType.ENDPOINT, "", self._lang, True))
            self._reset()
            return events

        # Иначе периодический партиал.
        if now_ms - self._last_partial_ms >= _PARTIAL_EVERY_MS and len(self._speech) > _SR // 2:
            self._last_partial_ms = now_ms
            text = await self._transcribe(final=False)
            if text:
                events.append(TranscriptEvent(TranscriptType.PARTIAL, text, self._lang, False))
        return events

    async def finalize(self) -> list[TranscriptEvent]:
        """Поток закрылся — доотдать накопленное как финал."""
        events: list[TranscriptEvent] = []
        if self._speech:
            text = await self._transcribe(final=True)
            if text:
                events.append(TranscriptEvent(TranscriptType.FINAL, text, self._lang, True))
        self._reset()
        return events

    # ── Внутреннее ───────────────────────────────────────────────────────────
    async def _vad_is_speech(self, frame) -> bool:
        import torch

        loop = asyncio.get_running_loop()

        def _run() -> float:
            with torch.no_grad():
                return self._engine._vad(torch.from_numpy(frame), _SR).item()

        prob = await loop.run_in_executor(None, _run)
        return prob >= 0.5

    async def _transcribe(self, *, final: bool) -> str:
        import numpy as np

        audio = np.asarray(self._speech, dtype=np.float32)
        if audio.size < _SR // 4:
            return ""
        loop = asyncio.get_running_loop()

        def _run() -> str:
            # Сериализуем доступ к модели: одна карта — один прогон за раз.
            segments, _ = self._engine._model.transcribe(
                audio, language=self._lang, beam_size=5 if final else 1,
                temperature=0.0, vad_filter=False, condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()

        async with self._engine._model_lock:
            return await loop.run_in_executor(None, _run)

    def _reset(self) -> None:
        self._speech = []
        self._silence_frames = 0
        self._speaking = False
        self._last_partial_ms = 0.0