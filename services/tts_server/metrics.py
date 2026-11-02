"""
Synctona TTS-сервис — метрики (Step 2).

Замеряет узлы из §2.1 ТЗ: TTFB, RTF, активные потоки, глубину очереди,
утилизацию, отказы/отмены. Экспортируется через /metrics (Prometheus).
Гистограммы TTFB/RTF — это то, чем калибруется max_streams_per_gpu (§5.4).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# TTFB: время до первого аудио-чанка. Бакеты вокруг бюджета 700 мс.
TTFB_MS = Histogram(
    "synctona_tts_ttfb_ms",
    "Time-to-first-audio-chunk (ms)",
    buckets=(50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2500, 5000),
)

# RTF: compute-seconds на секунду сгенерированного аудио.
RTF = Histogram(
    "synctona_tts_rtf",
    "Real-time factor (compute-s / audio-s)",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0),
)

ACTIVE_STREAMS = Gauge("synctona_tts_active_streams", "Активные TTS-потоки на пуле")
QUEUE_DEPTH = Gauge("synctona_tts_queue_depth", "Глубина очереди ожидания слота")
UTILIZATION = Gauge("synctona_tts_utilization", "active / capacity в пуле [0..1]")

JOBS_DONE = Counter("synctona_tts_jobs_total", "Завершённые TTS-джобы")
CANCELS = Counter("synctona_tts_cancels_total", "Отменённые джобы (barge-in и пр.)")
REJECTS = Counter("synctona_tts_rejects_total", "Отклонённые джобы (backpressure)")
ENGINE_ERRORS = Counter("synctona_tts_engine_errors_total", "Ошибки движка синтеза")