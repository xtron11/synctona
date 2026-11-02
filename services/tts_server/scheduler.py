"""
Synctona TTS-сервис — планировщик (Step 2).

Управляет ёмкостью GPU-пула и жизненным циклом джоб:
  • жёсткий потолок конкурентности = max_streams_per_gpu × serving_gpu_count;
  • очередь ожидания с отсечкой по возрасту (§2.2b ТЗ) — устаревшее не исполняем;
  • реестр активных джоб + кооперативная отмена по job_id (шаг 2 barge-in);
  • utilization() = active / capacity — источник для admission и лестницы деградации.

Интерфейс движка — TTSEngine из engine.py. Замена на батч-backend (Triton/NeMo)
не требует изменений здесь: планировщик оперирует джобами, а не токенами.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from tts_server.engine import SynthRequest, TTSEngine
from tts_server import metrics

logger = logging.getLogger("synctona.tts.scheduler")


class RejectedError(Exception):
    """Джоба отклонена планировщиком (перегрузка / устаревание в очереди)."""


@dataclass
class _Job:
    req: SynthRequest
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    enqueued_at: float = field(default_factory=time.monotonic)
    started_at: float = 0.0


class TTSScheduler:
    def __init__(
        self,
        engine: TTSEngine,
        *,
        max_streams_per_gpu: int,
        serving_gpu_count: int,
        queue_max: int,
        max_queue_age_ms: int,
        queue_wait_timeout_s: float = 1.5,
    ) -> None:
        self._engine = engine
        self._capacity = max(1, max_streams_per_gpu * serving_gpu_count)
        self._sem = asyncio.Semaphore(self._capacity)
        self._queue_max = queue_max
        self._max_queue_age_ms = max_queue_age_ms
        self._queue_wait_timeout_s = queue_wait_timeout_s

        self._active: dict[str, _Job] = {}
        self._waiting: int = 0
        self._lock = asyncio.Lock()

    # ── Ёмкость / метрики ────────────────────────────────────────────────────
    @property
    def capacity(self) -> int:
        return self._capacity

    def active_count(self) -> int:
        return len(self._active)

    def utilization(self) -> float:
        return min(1.0, len(self._active) / self._capacity)

    def _publish_gauges(self) -> None:
        metrics.ACTIVE_STREAMS.set(len(self._active))
        metrics.QUEUE_DEPTH.set(self._waiting)
        metrics.UTILIZATION.set(self.utilization())

    # ── Отмена ───────────────────────────────────────────────────────────────
    async def cancel(self, job_id: str) -> bool:
        job = self._active.get(job_id)
        if job is None:
            return False
        job.cancel_event.set()
        metrics.CANCELS.inc()
        logger.info("Отмена джобы %s (освобождение слота GPU)", job_id)
        return True

    # ── Основной путь: приём + стриминг ──────────────────────────────────────
    async def synthesize(self, req: SynthRequest) -> AsyncIterator[bytes]:
        """
        Полный цикл одной джобы: admission → ожидание слота → генерация → release.
        Бросает RejectedError, если пул перегружен или запрос устарел в очереди.
        """
        # Быстрая отсечка на входе: очередь переполнена.
        async with self._lock:
            if self._waiting >= self._queue_max:
                metrics.REJECTS.inc()
                raise RejectedError("Очередь TTS переполнена")
            self._waiting += 1
            self._publish_gauges()

        job = _Job(req=req)
        acquired = False
        try:
            # Ждём слот с таймаутом; при таймауте — отклоняем (backpressure).
            try:
                await asyncio.wait_for(self._sem.acquire(), timeout=self._queue_wait_timeout_s)
                acquired = True
            except asyncio.TimeoutError:
                metrics.REJECTS.inc()
                raise RejectedError("Нет свободного слота GPU (таймаут ожидания)")

            # Отсечка по возрасту: пока ждали, запрос мог устареть → не исполняем.
            age_ms = (time.monotonic() - job.enqueued_at) * 1000
            if age_ms > self._max_queue_age_ms:
                metrics.REJECTS.inc()
                raise RejectedError(f"Запрос устарел в очереди ({age_ms:.0f}ms)")

            async with self._lock:
                self._waiting -= 1
                self._active[req.job_id] = job
                job.started_at = time.monotonic()
                self._publish_gauges()

            async for chunk in self._run_job(job):
                yield chunk

        finally:
            async with self._lock:
                if req.job_id in self._active:
                    del self._active[req.job_id]
                elif not acquired:
                    self._waiting = max(0, self._waiting - 1)
                self._publish_gauges()
            if acquired:
                self._sem.release()

    async def _run_job(self, job: _Job) -> AsyncIterator[bytes]:
        req = job.req
        first_chunk = True
        audio_bytes = 0
        gen_start = time.monotonic()
        try:
            async for chunk in self._engine.stream(req, job.cancel_event):
                if job.cancel_event.is_set():
                    return
                if first_chunk:
                    ttfb = (time.monotonic() - job.started_at) * 1000
                    metrics.TTFB_MS.observe(ttfb)
                    first_chunk = False
                audio_bytes += len(chunk)
                yield chunk
        except asyncio.CancelledError:
            job.cancel_event.set()
            raise
        except Exception:
            metrics.ENGINE_ERRORS.inc()
            logger.exception("Ошибка движка на джобе %s", req.job_id)
            raise
        finally:
            # RTF = compute-время / длительность аудио (ключ для capacity §5.4).
            audio_s = (audio_bytes / 2) / max(1, self._engine.sample_rate)
            if audio_s > 0:
                rtf = (time.monotonic() - gen_start) / audio_s
                metrics.RTF.observe(rtf)
            metrics.JOBS_DONE.inc()