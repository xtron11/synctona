"""
Synctona TTS-сервис — движок (Step 2).

Адаптер над клон-TTS (Chatterbox v3 по умолчанию; Qwen3-TTS подключается тем же
интерфейсом). Отвечает за:
  • загрузку модели и прогрев;
  • кэш speaker-conditionals/латентов на голос (считаем один раз — §5.2 ТЗ);
  • стриминговую генерацию PCM-чанков с кооперативной отменой по cancel_event
    (это то, что освобождает слот GPU на шаге 2 barge-in).

ЧЕСТНО про батчинг: vanilla-Chatterbox держит conditionals на общем инстансе
модели и генерирует по одной фразе за раз на GPU. Настоящий inflight-batching —
это Triton/NeMo или движок с батч-декодом; интерфейс ниже спроектирован так, что
такой backend подключается заменой ChatterboxEngine на BatchedBackendEngine без
изменения планировщика и сервера. До этого честная ёмкость одной карты = число,
которое найдёт scripts/loadtest.py, а не «сколько угодно».
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Protocol, runtime_checkable

logger = logging.getLogger("synctona.tts.engine")


@dataclass
class SynthRequest:
    job_id: str
    text: str
    voice_id: str
    ref_wav_path: Optional[str]   # путь к образцу голоса (или None → generic/default)
    preset: str                   # "default" | "turbo" (быстрый деградационный пресет)
    sample_rate: int


@runtime_checkable
class TTSEngine(Protocol):
    sample_rate: int

    async def warmup(self) -> None: ...

    async def precompute_speaker(self, voice_id: str, ref_wav_path: Optional[str]) -> None:
        """Считает и кэширует conditionals голоса один раз (§5.2 ТЗ)."""
        ...

    def stream(
        self, req: SynthRequest, cancel_event: asyncio.Event
    ) -> AsyncIterator[bytes]:
        """Асинхронный поток PCM16-моно чанков для одного запроса."""
        ...


class ChatterboxEngine:
    """
    Конкретный движок на Chatterbox v3.

    Модель торчовая и не реентерабельна на одной карте, поэтому генерация
    сериализуется через _model_lock. Планировщик выше управляет конкурентностью
    и очередью; здесь — только корректная одиночная генерация + кэш + отмена.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        model_sample_rate: int = 24000,
        chunk_ms: int = 200,
        latent_cache_size: int = 256,
    ) -> None:
        self.sample_rate = model_sample_rate
        self._device = device
        self._chunk_ms = chunk_ms
        self._model = None
        self._model_lock = asyncio.Lock()
        # voice_id → ref_wav_path (conditionals пересчитываются перед generate под локом).
        self._latents: "OrderedDict[str, Optional[str]]" = OrderedDict()
        self._latent_cache_size = latent_cache_size
        self._loaded = asyncio.Event()

    # ── Загрузка / прогрев ───────────────────────────────────────────────────
    async def warmup(self) -> None:
        if self._loaded.is_set():
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_blocking)
        # Прогревочная генерация (компиляция графа/кэши CUDA).
        warm = SynthRequest(
            job_id="__warmup__", text="warm up", voice_id="default",
            ref_wav_path=None, preset="default", sample_rate=self.sample_rate,
        )
        async for _ in self.stream(warm, asyncio.Event()):
            pass
        self._loaded.set()
        logger.info("ChatterboxEngine прогрет (device=%s, sr=%d)", self._device, self.sample_rate)

    def _load_blocking(self) -> None:
        # Изолированный импорт: сервис TTS не тащит зависимости в agent-worker.
        from chatterbox.tts import ChatterboxTTS  # type: ignore

        self._model = ChatterboxTTS.from_pretrained(device=self._device)
        # Синхронизируем sr с моделью, если она задаёт свой.
        self.sample_rate = int(getattr(self._model, "sr", self.sample_rate))

    def remove_speaker(self, voice_id: str) -> None:
        self._latents.pop(voice_id, None)
        
    # ── Кэш голоса ───────────────────────────────────────────────────────────
    async def precompute_speaker(self, voice_id: str, ref_wav_path: Optional[str]) -> None:
        if voice_id in self._latents:
            self._latents.move_to_end(voice_id)
            return
        self._latents[voice_id] = ref_wav_path
        # LRU-эвикция.
        while len(self._latents) > self._latent_cache_size:
            evicted, _ = self._latents.popitem(last=False)
            logger.debug("LRU-эвикция латентов голоса %s", evicted)
        # Опционально: прогреть conditionals заранее, если API это позволяет.
        if ref_wav_path and self._model is not None:
            prepare = getattr(self._model, "prepare_conditionals", None)
            if callable(prepare):
                loop = asyncio.get_running_loop()
                async with self._model_lock:
                    with _suppress("prepare_conditionals"):
                        await loop.run_in_executor(None, prepare, ref_wav_path)

    # ── Генерация ────────────────────────────────────────────────────────────
    async def stream(
        self, req: SynthRequest, cancel_event: asyncio.Event
    ) -> AsyncIterator[bytes]:
        if self._model is None:
            raise RuntimeError("Движок не прогрет (warmup не вызван).")
        if cancel_event.is_set():
            return

        loop = asyncio.get_running_loop()
        ref = self._latents.get(req.voice_id, req.ref_wav_path)

        # Пытаемся нативный стриминг, если движок его отдаёт; иначе — full-generate
        # с последующей нарезкой (TTFB тогда = полное время генерации фразы).
        native = getattr(self._model, "generate_stream", None)
        async with self._model_lock:
            if callable(native):
                async for chunk in self._native_stream(native, req, ref, cancel_event, loop):
                    yield chunk
            else:
                async for chunk in self._chunked_fallback(req, ref, cancel_event, loop):
                    yield chunk

    async def _native_stream(self, native, req, ref, cancel_event, loop) -> AsyncIterator[bytes]:
        # Форк chatterbox-streaming: generate_stream(...) -> генератор, отдающий на
        # каждом шаге КОРТЕЖ (audio_tensor, metrics). Берём [0]. Каждый шаг крутим
        # в executor, чтобы не блокировать event loop и оперативно ловить отмену.
        def _make():
            kwargs = {"chunk_size": 25, "print_metrics": False}
            if ref:
                kwargs["audio_prompt_path"] = ref
            kwargs["cfg_weight"] = 0.3
            return native(req.text, **kwargs)

        sentinel = object()

        def _next(g):
            try:
                return next(g)
            except StopIteration:
                return sentinel

        gen = await loop.run_in_executor(None, _make)
        while True:
            if cancel_event.is_set():
                logger.debug("Отмена во время native-стрима (job=%s)", req.job_id)
                return
            item = await loop.run_in_executor(None, _next, gen)
            if item is sentinel:
                return
            audio = item[0] if isinstance(item, (tuple, list)) else item
            pcm = self._to_pcm16(audio)
            if pcm:
                yield pcm

    async def _chunked_fallback(self, req, ref, cancel_event, loop) -> AsyncIterator[bytes]:
        def _generate():
            kwargs = {}
            if ref:
                kwargs["audio_prompt_path"] = ref
            if req.preset == "turbo":
                kwargs["cfg_weight"] = 0.3
            return self._model.generate(req.text, **kwargs)

        wav = await loop.run_in_executor(None, _generate)
        pcm = self._to_pcm16(wav)
        # Нарезаем на ~chunk_ms, чтобы клиент/плеер получал поток, а не один блок.
        bytes_per_chunk = int(self.sample_rate * self._chunk_ms / 1000) * 2  # int16 = 2 байта
        for i in range(0, len(pcm), bytes_per_chunk):
            if cancel_event.is_set():
                logger.debug("Отмена во время fallback-стрима (job=%s)", req.job_id)
                return
            yield pcm[i:i + bytes_per_chunk]
            await asyncio.sleep(0)  # уступаем цикл, чтобы отмена срабатывала оперативно

    # ── Конвертация float32 [-1,1] → PCM16 bytes ─────────────────────────────
    @staticmethod
    def _to_pcm16(tensor) -> bytes:
        import numpy as np

        if tensor is None:
            return b""
        
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        arr = np.asarray(arr, dtype=np.float32).flatten()
        if arr.size == 0:
            return b""
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16).tobytes()


class _suppress:
    """contextmanager-логгер для необязательных операций."""

    def __init__(self, what: str) -> None:
        self._what = what

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            logger.warning("Необязательная операция '%s' не удалась: %s", self._what, exc)
        return True