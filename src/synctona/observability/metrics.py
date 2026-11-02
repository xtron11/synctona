"""
Synctona — метрики агента (Step 5).

Prometheus-инструменты со стороны оркестратора. Ключевая — E2E_FIRST_AUDIO_MS:
это сам SLO из ТЗ (t0 → первый звук), который нельзя померить в сервисах, только
здесь. Экспортируется отдельным HTTP-портом (Prometheus его скрейпит).

Это НЕ тот metrics.py, что в services/*/ — те про сервисы, этот про агента.
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger("synctona.metrics")

# ── SLO: сквозная задержка от endpoint до первого звука ──────────────────────
E2E_FIRST_AUDIO_MS = Histogram(
    "synctona_agent_e2e_first_audio_ms",
    "Endpoint → первый аудио-кадр перевода (ms). Это и есть SLO.",
    buckets=(100, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3500, 5000, 8000),
)

# ── Турны и состояние ────────────────────────────────────────────────────────
TURNS_TOTAL = Counter("synctona_agent_turns_total", "Обработанные турны перевода")
BARGE_IN_TOTAL = Counter("synctona_agent_barge_in_total", "Срабатывания barge-in")
ACTIVE_SESSIONS = Gauge("synctona_agent_active_sessions", "Активные сессии агента")

# ── Admission / деградация ───────────────────────────────────────────────────
ADMISSION_ADMITS = Counter("synctona_agent_admission_admits_total", "Принятые сессии")
ADMISSION_REJECTS = Counter("synctona_agent_admission_rejects_total", "Отклонённые сессии (перегрузка)")
DEGRADATION_LEVEL = Gauge(
    "synctona_agent_degradation_level", "Активный уровень деградации (0=FULL..3=TEXT_ONLY)"
)

# ── Ошибки стадий (для алертов) ──────────────────────────────────────────────
STAGE_ERRORS = Counter(
    "synctona_agent_stage_errors_total", "Ошибки стадий каскада", ["stage"]
)

_started = False


def start_metrics_server(port: int = 9100) -> None:
    """Поднимает HTTP-экспортёр Prometheus (idempotent). Вызывать один раз на процесс."""
    global _started
    if _started:
        return
    try:
        start_http_server(port)
        _started = True
        logger.info("Prometheus-экспортёр агента на :%d/metrics", port)
    except Exception:
        logger.exception("Не удалось поднять экспортёр метрик на :%d", port)