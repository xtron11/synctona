"""
Synctona — трейсинг агента (Step 5).

Тонкая обёртка над OpenTelemetry. setup_tracing() поднимает OTLP-экспортёр (спаны
уходят в Jaeger/collector), span() — контекст-менеджер для инструментирования
стадий каскада. Если OTEL_EXPORTER_OTLP_ENDPOINT не задан или SDK не установлен —
всё превращается в no-op, агент работает как обычно.

Инструментирование конкретных стадий (asr_finalize → nmt → tts_ttfb) — инкремент:
достаточно обернуть их в `with span("nmt"): ...`. Здесь дан каркас.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator, Optional

logger = logging.getLogger("synctona.tracing")

_tracer = None


def setup_tracing(service_name: str, endpoint: Optional[str]) -> None:
    """Инициализирует OTLP-трейсинг. Без endpoint — no-op."""
    global _tracer
    if not endpoint:
        logger.info("OTEL endpoint не задан — трейсинг выключен.")
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
        logger.info("OTLP-трейсинг включён → %s", endpoint)
    except Exception:
        logger.exception("Не удалось инициализировать трейсинг — продолжаю без него.")
        _tracer = None


@contextlib.contextmanager
def span(name: str, **attrs) -> Iterator[None]:
    """Оборачивает участок кода в спан. No-op, если трейсинг не поднят."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            with contextlib.suppress(Exception):
                sp.set_attribute(k, v)
        yield