"""
Synctona — центральная конфигурация (Step 1).

Все параметры из architecture spec собраны здесь и грузятся из окружения через
pydantic-settings. Ничего не хардкодим по месту — модули читают настройки отсюда.

Переменные окружения: префикс SYNCTONA_, вложенность через "__".
Пример: SYNCTONA_LIVEKIT__URL, SYNCTONA_TTS_POOL__MAX_STREAMS_PER_GPU.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Лестница деградации (Backpressure) — §2.2(c) ТЗ
# ─────────────────────────────────────────────────────────────────────────────
class DegradationAction(str, Enum):
    """Действие уровня деградации, применяемое к TTS-пути при росте нагрузки."""

    FULL = "full"                 # полный клон-голос, полное качество
    FAST_TTS = "fast_tts"         # быстрый пресет TTS, greedy NMT
    GENERIC_VOICE = "generic"     # generic-голос вместо клонирования (дешевле по GPU)
    TEXT_ONLY = "text_only"       # только субтитры, TTS отключён


class DegradationLevel(BaseSettings):
    """Один уровень лестницы: диапазон утилизации TTS-пула → действие."""

    name: str
    util_min: float               # включительно
    util_max: float               # не включительно (1.01 для верхней ступени)
    action: DegradationAction
    note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Секции конфигурации
# ─────────────────────────────────────────────────────────────────────────────
class LiveKitSettings(BaseSettings):
    url: str = Field(default="ws://localhost:7880")
    api_key: str = Field(default="devkey")
    api_secret: str = Field(default="secret")
    agent_name: str = Field(default="synctona-translator")
    # Явный dispatch: агент стартует только по запросу сессии, не заходит в каждую комнату.
    explicit_dispatch: bool = True
    # Сколько одновременных Job тянет один worker-процесс (не путать с TTS-конкурентностью).
    max_jobs_per_worker: int = 32
    # Секунды прогрева AEC на входе сессии (echo до готовности AEC не должно триггерить barge-in).
    aec_warmup_s: float = 3.0


class SLOSettings(BaseSettings):
    e2e_first_audio_p95_s: float = 2.5     # целевой SLO из ТЗ
    e2e_first_audio_p99_s: float = 3.5
    tts_ttfb_budget_ms: int = 700          # порог TTFB, после которого масштабируемся / деградируем


class EndpointingSettings(BaseSettings):
    """
    §0 ТЗ: endpoint-задержка — несжимаемый пол турна. В STT-режиме min_delay
    аддитивен к endpointing самого провайдера, поэтому держим его низким и
    полагаемся на семантический turn-detector.
    """

    mode: Literal["fixed", "dynamic"] = "fixed"
    min_endpointing_delay_s: float = 0.20
    max_endpointing_delay_s: float = 6.0
    # Барж-ин: не прерывать озвучку на кашель/шум — нужен минимум слов/длительности.
    interrupt_min_words: int = 2
    interrupt_speech_duration_s: float = 0.20
    # Семантический turn-detector (transformer) вместо голого VAD-молчания.
    use_semantic_turn_detector: bool = True
    vad_model: str = "silero"


class QueueSettings(BaseSettings):
    """§2.2(b) ТЗ: очереди фиксированной длины + отсечка по возрасту."""

    asr_queue_max: int = 64
    nmt_queue_max: int = 64
    tts_queue_max: int = 32
    # Задача старше этого возраста в очереди дропается (исполнять устаревшее нельзя).
    max_task_age_ms: int = 800


class TTSPoolSettings(BaseSettings):
    """§5 ТЗ: параметры TTS-пула. max_streams_per_gpu калибруется load-тестом (§5.4)."""

    engine: Literal["chatterbox_v3", "qwen3_tts"] = "chatterbox_v3"
    endpoint: str = Field(default="http://tts-pool:8001")   # Triton/NeMo батчинг-сервис
    sample_rate: int = 24000
    # Кол-во GPU, реально обслуживающих TTS (без учёта N+1 резерва).
    serving_gpu_count: int = 2
    # ГЛАВНЫЙ параметр ёмкости. Значение по умолчанию — гипотеза до калибровки.
    max_streams_per_gpu: int = 16
    target_utilization: float = 0.70                        # держим util ниже ради p95
    ttfb_alarm_ms: int = 700
    # Быстрый деградационный пресет и generic-голос (для лестницы backpressure).
    fast_preset: str = "turbo"
    generic_voice_id: str = "synctona_generic_en"
    default_language: str = "en"
    # Таймаут одной TTS-джобы (жёсткий предохранитель).
    job_timeout_s: float = 20.0


class AdmissionSettings(BaseSettings):
    """§2.2(a) ТЗ: admission control на старте сессии."""

    # Порог утилизации TTS-пула, выше которого новые сессии не принимаются.
    gpu_util_admit_threshold: float = 0.70
    # Сообщение пользователю при мягком отказе.
    reject_message: str = "Сервис сейчас перегружен. Попробуйте переподключиться через минуту."
    retry_after_s: int = 60


class NMTSettings(BaseSettings):
    """§4 ТЗ: перевод целыми предложениями, guard длинных фраз."""

    endpoint: str = Field(default="http://nmt:8002")
    model: str = "Helsinki-NLP/opus-mt-mul-en"   # мульти→en; на паре можно узкий opus-mt
    beam_size: int = 4
    greedy_on_degradation: bool = True            # при деградации beam→1
    # Единица перевода — предложение. Аномально длинные режем по клаузам.
    max_sentence_words: int = 14
    clause_split_chars: str = ",;:—"
    # Rolling-контекст для связности (последние N предложений как контекст NMT).
    rolling_context_sentences: int = 2
    request_timeout_s: float = 4.0


class ChunkingSettings(BaseSettings):
    """§4 ТЗ: pacing и склейка на клиенте/выходе."""

    jitter_buffer_ms: int = 200          # клиентский буфер (150–250 по ТЗ)
    sentence_join_gap_ms: int = 120      # микропауза на стыке предложений
    enable_speculative_translation: bool = False  # по умолчанию sentence-final


class ObservabilitySettings(BaseSettings):
    otel_endpoint: str = Field(default="http://otel-collector:4317")
    prometheus_port: int = 9464
    service_name: str = "synctona-agent"
    log_level: str = "INFO"


class RuntimeSettings(BaseSettings):
    warmup_retries: int = 3
    warmup_backoff_s: float = 2.0
    # Сколько ждём graceful-завершения задач при отмене, прежде чем жёстко снять.
    task_cancel_grace_s: float = 2.0
    default_source_lang: str = "ru"
    default_target_lang: str = "en"
    # Поддерживаемые пары (для мульти-язычного масштабирования).
    supported_pairs: tuple[tuple[str, str], ...] = (("ru", "en"), ("en", "ru"))


# ─────────────────────────────────────────────────────────────────────────────
# Корневые настройки
# ─────────────────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SYNCTONA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    livekit: LiveKitSettings = Field(default_factory=LiveKitSettings)
    slo: SLOSettings = Field(default_factory=SLOSettings)
    endpointing: EndpointingSettings = Field(default_factory=EndpointingSettings)
    queues: QueueSettings = Field(default_factory=QueueSettings)
    tts_pool: TTSPoolSettings = Field(default_factory=TTSPoolSettings)
    admission: AdmissionSettings = Field(default_factory=AdmissionSettings)
    nmt: NMTSettings = Field(default_factory=NMTSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    # ── Производные величины ──────────────────────────────────────────────────
    @computed_field  # type: ignore[prop-decorator]
    @property
    def global_max_concurrent_tts(self) -> int:
        """
        §5.1 ТЗ: глобальный лимит одновременных TTS-потоков.
            capacity = max_streams_per_gpu × serving_gpu_count × target_utilization
        Именно это число заряжается в семафор admission control.
        """
        cap = (
            self.tts_pool.max_streams_per_gpu
            * self.tts_pool.serving_gpu_count
            * self.tts_pool.target_utilization
        )
        return max(1, int(cap))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def degradation_ladder(self) -> list[DegradationLevel]:
        """§2.2(c) ТЗ. Диапазоны утилизации TTS-пула → действие."""
        return [
            DegradationLevel(name="L0", util_min=0.00, util_max=0.70,
                             action=DegradationAction.FULL,
                             note="полное качество, клон-голос"),
            DegradationLevel(name="L1", util_min=0.70, util_max=0.85,
                             action=DegradationAction.FAST_TTS,
                             note="быстрый TTS-пресет, greedy NMT"),
            DegradationLevel(name="L2", util_min=0.85, util_max=0.95,
                             action=DegradationAction.GENERIC_VOICE,
                             note="generic-голос вместо клонирования"),
            DegradationLevel(name="L3", util_min=0.95, util_max=1.01,
                             action=DegradationAction.TEXT_ONLY,
                             note="только субтитры, TTS off"),
        ]

    def level_for_utilization(self, utilization: float) -> DegradationLevel:
        """Возвращает активный уровень деградации для текущей утилизации пула."""
        u = min(max(utilization, 0.0), 1.0)
        for level in self.degradation_ladder:
            if level.util_min <= u < level.util_max:
                return level
        return self.degradation_ladder[-1]

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if not (0.0 < self.tts_pool.target_utilization <= 1.0):
            raise ValueError("tts_pool.target_utilization должен быть в (0, 1].")
        if self.tts_pool.max_streams_per_gpu < 1:
            raise ValueError("tts_pool.max_streams_per_gpu должен быть >= 1.")
        if self.endpointing.min_endpointing_delay_s < 0:
            raise ValueError("min_endpointing_delay_s не может быть отрицательным.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Синглтон настроек. Кэшируется, чтобы не перечитывать окружение на каждый импорт."""
    return Settings()