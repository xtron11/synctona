"""
Synctona — оркестратор сессии (Step 1).

Реализует поверх LiveKit Agents 1.x:
  • Конечный автомат турна: LISTENING → TRANSLATING → SPEAKING → BARGE_IN.
  • Атомарную обработку barge-in строго из 5 шагов (§3.2 ТЗ).
  • Пайплайнинг предложений (§4 ТЗ): следующее предложение переводится/озвучивается,
    пока текущее ещё играет.

Модель интеграции с LiveKit 1.x:
  • Кастомный Agent переопределяет node-хуки: llm_node → NMT, tts_node → наш клон-TTS.
  • Barge-in фреймворк детектит сам (turn-detector + interrupt_min_words) и бросает
    asyncio.CancelledError в наши генераторы. Это версионно-надёжная точка, куда
    навешивается очистка НАШИХ ресурсов (GPU-джоба TTS, хвост NMT), о которых
    фреймворк не знает.

ВНИМАНИЕ по версиям SDK: сигнатуры node-хуков и имена event'ов в livekit-agents
эволюционируют. Все прямые вызовы SDK изолированы в маленьких _lk_* хелперах и
обёрнуты guard'ами — при обновлении SDK правится только там, оркестрационная
логика остаётся нетронутой.

Конкретные клиенты ASR/NMT/TTS (self-hosted) приходят на Step 2/3. Здесь их
контракты объявлены как Protocol; оркестратор полностью реализован против них.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterable, Optional, Protocol, runtime_checkable

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    ModelSettings,
    cli,
)
from livekit.agents import llm as lk_llm

from synctona.config import DegradationAction, Settings, get_settings
from synctona.clients.asr_adapter import SynctonaSTT
from synctona.clients.speaker_registry import RedisSpeakerRegistry
from synctona.observability import metrics, tracing

logger = logging.getLogger("synctona.agent")


# ─────────────────────────────────────────────────────────────────────────────
# Контракты внешних сервисов (реализация — Step 2/3). Оркестратор зависит только
# от этих интерфейсов, что позволяет тестировать его в изоляции.
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class NMTClient(Protocol):
    async def warmup(self) -> None: ...

    async def translate(
        self, text: str, src: str, tgt: str, *, beam_size: int, context: list[str]
    ) -> str:
        """Переводит ОДНО предложение целиком (§4 ТЗ)."""
        ...


@runtime_checkable
class TTSJobHandle(Protocol):
    """Дескриптор запущенной TTS-джобы для отмены на удалённом GPU-сервисе."""

    job_id: str

    async def cancel(self) -> None:
        """Сигнал удалённому TTS-пулу прекратить генерацию и освободить слот GPU."""
        ...


@runtime_checkable
class TTSClient(Protocol):
    async def warmup(self) -> None: ...

    async def utilization(self) -> float:
        """Текущая утилизация TTS-пула [0..1] для admission/деградации."""
        ...

    def synthesize(
        self,
        text: str,
        *,
        session_id: str,
        voice: str,
        preset: str,
        sample_rate: int,
    ) -> "TTSStream":
        """
        Возвращает объект-стрим: async-итератор аудио-фреймов + .handle для отмены.
        Не корутина — синхронно отдаёт стрим, чтобы можно было сразу забрать handle.
        """
        ...


@runtime_checkable
class TTSStream(Protocol):
    handle: TTSJobHandle

    def __aiter__(self) -> AsyncIterable[rtc.AudioFrame]: ...


# ─────────────────────────────────────────────────────────────────────────────
# Конечный автомат турна
# ─────────────────────────────────────────────────────────────────────────────
class TurnState(Enum):
    LISTENING = auto()     # ждём/принимаем речь пользователя
    TRANSLATING = auto()   # NMT переводит финализированное предложение
    SPEAKING = auto()      # TTS играет перевод
    BARGE_IN = auto()      # идёт атомарная очистка после прерывания
    CLOSED = auto()        # сессия завершается


# Разрешённые переходы. Любой другой переход логируется как аномалия.
_ALLOWED_TRANSITIONS: dict[TurnState, set[TurnState]] = {
    TurnState.LISTENING: {TurnState.TRANSLATING, TurnState.CLOSED},
    TurnState.TRANSLATING: {TurnState.SPEAKING, TurnState.BARGE_IN, TurnState.LISTENING, TurnState.CLOSED},
    TurnState.SPEAKING: {TurnState.BARGE_IN, TurnState.LISTENING, TurnState.CLOSED},
    TurnState.BARGE_IN: {TurnState.LISTENING, TurnState.CLOSED},
    TurnState.CLOSED: set(),
}


@dataclass
class SessionRuntime:
    """Изменяемое состояние одной сессии."""

    session_id: str
    src_lang: str
    tgt_lang: str
    speaker_voice: str
    state: TurnState = TurnState.LISTENING

    # Активные ресурсы текущего турна (для отмены при barge-in).
    active_tts_handle: Optional[TTSJobHandle] = None
    active_nmt_task: Optional[asyncio.Task] = None
    pending_sentences: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Синхронизация.
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    barge_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    barge_in_seq: int = 0            # монотонный счётчик прерываний (для идемпотентности)

    # Метрики турна.
    turn_t0: float = 0.0             # момент endpoint (начало SLO-отсчёта)
    ttfb_reported: bool = False

    # Rolling-контекст для связности перевода (§4 ТЗ).
    context_history: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты отмены (utils/cancellation.py в репо; здесь для самодостаточности Step 1)
# ─────────────────────────────────────────────────────────────────────────────
async def cancel_and_wait(task: Optional[asyncio.Task], grace_s: float) -> None:
    """Кооперативно отменяет задачу и ждёт её завершения в пределах grace."""
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=grace_s)


# ─────────────────────────────────────────────────────────────────────────────
# Основной агент-оркестратор
# ─────────────────────────────────────────────────────────────────────────────
class SynctonaTranslatorAgent(Agent):
    """
    Кастомный Agent: llm_node переопределён под NMT, tts_node — под клон-TTS.
    Состояние турна и barge-in управляются явно поверх событий фреймворка.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        nmt: NMTClient,
        tts: TTSClient,
        runtime: SessionRuntime,
    ) -> None:
        # Инструкции агенту не нужны (это переводчик, не LLM-ассистент), но базовый
        # Agent требует instructions — задаём пустышку, node-хуки всё равно перекрыты.
        super().__init__(instructions="Synctona speech translator. Nodes are overridden.")
        self._s = settings
        self._nmt = nmt
        self._tts = tts
        self._rt = runtime
        self._session: Optional[AgentSession] = None

    # ── Управление состоянием ────────────────────────────────────────────────
    async def _transition(self, new: TurnState) -> None:
        async with self._rt.state_lock:
            cur = self._rt.state
            if cur is new:
                return
            if new not in _ALLOWED_TRANSITIONS.get(cur, set()):
                logger.warning("Аномальный переход состояния %s → %s (session=%s)",
                               cur.name, new.name, self._rt.session_id)
            self._rt.state = new
            logger.debug("Состояние: %s → %s (session=%s)", cur.name, new.name, self._rt.session_id)

    # ── Жизненный цикл ───────────────────────────────────────────────────────
    async def on_enter(self) -> None:
        """Агент стал активным в сессии."""
        self._session = self.session
        await self._transition(TurnState.LISTENING)
        self._lk_register_interruption_handler()
        logger.info("Сессия готова: %s (%s→%s, voice=%s)",
                    self._rt.session_id, self._rt.src_lang, self._rt.tgt_lang,
                    self._rt.speaker_voice)

    async def on_exit(self) -> None:
        await self._transition(TurnState.CLOSED)
        await self._cleanup_turn_resources(reason="session_exit")
        logger.info("Сессия завершена: %s", self._rt.session_id)

    # ── NMT-нода (бывший llm_node) ───────────────────────────────────────────
    async def llm_node(
        self,
        chat_ctx: lk_llm.ChatContext,
        tools: list,
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        """
        Вместо LLM — перевод. Берём последнюю реплику пользователя, режем на
        предложения и переводим ПО ОДНОМУ, отдавая переводы фреймворку по мере
        готовности (пайплайнинг: TTS начинает озвучку S1, пока переводится S2).
        """
        self._rt.turn_t0 = time.monotonic()   # endpoint зафиксирован фреймворком
        self._rt.ttfb_reported = False
        await self._transition(TurnState.TRANSLATING)

        source_text = self._extract_last_user_text(chat_ctx)
        if not source_text:
            await self._transition(TurnState.LISTENING)
            return

        sentences = self._segment_sentences(source_text)
        beam = 1 if self._degraded_greedy() else self._s.nmt.beam_size

        try:
            for sentence in sentences:
                # Отдельная задача на предложение → её можно отменить при barge-in,
                # не трогая уже отданные фреймворку переводы.
                self._rt.active_nmt_task = asyncio.create_task(
                    self._translate_one(sentence, beam)
                )
                try:
                    translated = await self._rt.active_nmt_task
                except asyncio.CancelledError:
                    logger.debug("NMT-предложение отменено (barge-in), session=%s",
                                 self._rt.session_id)
                    raise
                except Exception:
                    logger.exception("Ошибка NMT, предложение пропущено (session=%s)",
                                     self._rt.session_id)
                    continue
                finally:
                    self._rt.active_nmt_task = None

                if translated:
                    self._push_context(sentence)
                    yield translated
        except asyncio.CancelledError:
            # Шаг 4 barge-in (сброс хвоста NMT) исполняется в _handle_barge_in;
            # здесь просто корректно выходим из генератора.
            self._drain_pending_sentences()
            raise

    async def _translate_one(self, sentence: str, beam: int) -> str:
        return await asyncio.wait_for(
            self._nmt.translate(
                sentence,
                self._rt.src_lang,
                self._rt.tgt_lang,
                beam_size=beam,
                context=list(self._rt.context_history),
            ),
            timeout=self._s.nmt.request_timeout_s,
        )

    # ── TTS-нода ─────────────────────────────────────────────────────────────
    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        """
        Наш клон-TTS. На вход — поток переведённых предложений из llm_node.
        Для каждого запускаем стриминговую синтезацию, регистрируем handle
        (для отмены GPU-джобы при barge-in) и меряем TTFB.
        """
        # Деградация: text-only режим — не синтезируем, отдаём субтитры (шлём как
        # data-сообщение) и завершаем без аудио.
        if await self._degradation_action() is DegradationAction.TEXT_ONLY:
            async for sentence in text:
                await self._lk_send_subtitle(sentence)
            return

        preset, voice = await self._tts_preset_and_voice()
        await self._transition(TurnState.SPEAKING)

        async for sentence in text:
            if not sentence.strip():
                continue
            stream = self._tts.synthesize(
                sentence,
                session_id=self._rt.session_id,
                voice=voice,
                preset=preset,
                sample_rate=self._s.tts_pool.sample_rate,
            )
            self._rt.active_tts_handle = stream.handle
            try:
                async for frame in self._iter_with_timeout(stream, self._s.tts_pool.job_timeout_s):
                    self._maybe_report_ttfb()
                    yield frame
            except asyncio.CancelledError:
                # Barge-in во время озвучки: гарантированно освобождаем GPU-слот.
                await self._safe_cancel_tts_handle(stream.handle)
                raise
            except asyncio.TimeoutError:
                logger.warning("TTS job timeout, предложение оборвано (session=%s)",
                               self._rt.session_id)
                await self._safe_cancel_tts_handle(stream.handle)
            finally:
                self._rt.active_tts_handle = None

            # Микропауза на стыке предложений (§4 pacing).
            gap = self._silence_frame(self._s.chunking.sentence_join_gap_ms)
            if gap is not None:
                yield gap

        await self._transition(TurnState.LISTENING)

    # ── Barge-in: атомарные 5 шагов (§3.2 ТЗ) ────────────────────────────────
    async def _handle_barge_in(self, reason: str) -> None:
        """
        Идемпотентно (через barge_lock + seq) выполняет строго 5 шагов.
        Шаги 1,3,5 — транспортные, делегируются фреймворку; шаги 2,4 — наши
        self-hosted ресурсы, которые фреймворк не знает.
        """
        async with self._rt.barge_lock:
            self._rt.barge_in_seq += 1
            seq = self._rt.barge_in_seq

            # Запись метрики прерывания
            metrics.BARGE_IN_TOTAL.inc()

            await self._transition(TurnState.BARGE_IN)
            logger.info("BARGE-IN #%d (reason=%s, session=%s)", seq, reason, self._rt.session_id)

            # Шаг 1: остановить плеер (фреймворк уже гасит речь при interrupt; вызываем идемпотентно).
            await self._lk_stop_agent_playout()

            # Шаг 2: отменить TTS-джобу → освободить слот GPU (НАШ ресурс).
            handle = self._rt.active_tts_handle
            if handle is not None:
                await self._safe_cancel_tts_handle(handle)
                self._rt.active_tts_handle = None

            # Шаг 3: очистить очередь исходящего аудио в SFU/выходном буфере.
            await self._lk_flush_output_buffers()

            # Шаг 4: сбросить хвост NMT — отменить текущую задачу и очистить очередь предложений.
            await cancel_and_wait(self._rt.active_nmt_task, self._s.runtime.task_cancel_grace_s)
            self._rt.active_nmt_task = None
            self._drain_pending_sentences()

            # Шаг 5: re-endpoint ASR — начать новый пользовательский турн с чистого VAD.
            await self._lk_reset_endpoint()

            await self._transition(TurnState.LISTENING)
            logger.info("BARGE-IN #%d завершён, LISTENING (session=%s)", seq, self._rt.session_id)

    async def _cleanup_turn_resources(self, reason: str) -> None:
        """Полная зачистка ресурсов турна (при exit / фатальной ошибке)."""
        with contextlib.suppress(Exception):
            if self._rt.active_tts_handle is not None:
                await self._safe_cancel_tts_handle(self._rt.active_tts_handle)
                self._rt.active_tts_handle = None
        await cancel_and_wait(self._rt.active_nmt_task, self._s.runtime.task_cancel_grace_s)
        self._rt.active_nmt_task = None
        self._drain_pending_sentences()

    # ── Хелперы NMT/сегментации/контекста ────────────────────────────────────
    @staticmethod
    def _extract_last_user_text(chat_ctx: lk_llm.ChatContext) -> str:
        """Достаёт текст последней реплики пользователя из контекста фреймворка."""
        for item in reversed(getattr(chat_ctx, "items", []) or []):
            if getattr(item, "role", None) == "user":
                content = getattr(item, "content", None)
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, (list, tuple)):
                    parts = [c for c in content if isinstance(c, str)]
                    return " ".join(parts).strip()
        return ""

    def _segment_sentences(self, text: str) -> list[str]:
        """
        Сегментация на предложения + guard длинных фраз (§4 ТЗ).
        В Step 2 заменяется на pipeline/sentence_segmenter.py (pysbd/семантика);
        здесь — детерминированная базовая реализация, чтобы оркестратор был рабочим.
        """
        import re

        raw = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
        max_words = self._s.nmt.max_sentence_words
        split_chars = set(self._s.nmt.clause_split_chars)

        result: list[str] = []
        for sent in raw:
            if len(sent.split()) <= max_words:
                result.append(sent)
                continue
            # Режем по безопасным границам клауз, чтобы ограничить TTFB.
            buf: list[str] = []
            for token in sent.split():
                buf.append(token)
                if token and token[-1] in split_chars and len(buf) >= 3:
                    result.append(" ".join(buf))
                    buf = []
            if buf:
                result.append(" ".join(buf))
        return result

    def _push_context(self, sentence: str) -> None:
        self._rt.context_history.append(sentence)
        limit = self._s.nmt.rolling_context_sentences
        if len(self._rt.context_history) > limit:
            self._rt.context_history = self._rt.context_history[-limit:]

    def _drain_pending_sentences(self) -> None:
        q = self._rt.pending_sentences
        while not q.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                q.get_nowait()

    # ── Хелперы TTS/деградации/метрик ────────────────────────────────────────
    def _degraded_greedy(self) -> bool:
        return self._s.nmt.greedy_on_degradation and self._rt.state is TurnState.BARGE_IN

    async def _degradation_action(self) -> DegradationAction:
        try:
            util = await self._tts.utilization()
        except Exception:
            logger.warning("Не удалось получить утилизацию TTS-пула, считаем FULL")
            return DegradationAction.FULL
        return self._s.level_for_utilization(util).action

    async def _tts_preset_and_voice(self) -> tuple[str, str]:
        action = await self._degradation_action()
        if action is DegradationAction.GENERIC_VOICE:
            return self._s.tts_pool.fast_preset, self._s.tts_pool.generic_voice_id
        if action is DegradationAction.FAST_TTS:
            return self._s.tts_pool.fast_preset, self._rt.speaker_voice
        return "default", self._rt.speaker_voice

    async def _tts_preset_only(self) -> str:
        preset, _ = await self._tts_preset_and_voice()
        return preset

    def _maybe_report_ttfb(self) -> None:
        if self._rt.ttfb_reported:
            return
        self._rt.ttfb_reported = True
        ttfb_ms = (time.monotonic() - self._rt.turn_t0) * 1000.0

        # Запись SLO в Prometheus
        metrics.E2E_FIRST_AUDIO_MS.observe(ttfb_ms)

        # В Step 2 уходит в observability/metrics.py (Prometheus histogram + OTel span).
        logger.info("TTFB=%.0fms (session=%s, budget=%dms)",
                    ttfb_ms, self._rt.session_id, self._s.slo.tts_ttfb_budget_ms)
        if ttfb_ms > self._s.slo.tts_ttfb_budget_ms:
            logger.warning("TTFB превысил бюджет (session=%s)", self._rt.session_id)

    async def _safe_cancel_tts_handle(self, handle: TTSJobHandle) -> None:
        with contextlib.suppress(Exception):
            await handle.cancel()

    async def _iter_with_timeout(
        self, stream: TTSStream, timeout_s: float
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Итерирует TTS-стрим с таймаутом на ПЕРВЫЙ и последующие фреймы."""
        it = stream.__aiter__()
        while True:
            try:
                frame = await asyncio.wait_for(it.__anext__(), timeout=timeout_s)
            except StopAsyncIteration:
                return
            yield frame

    def _silence_frame(self, gap_ms: int) -> Optional[rtc.AudioFrame]:
        if gap_ms <= 0:
            return None
        sr = self._s.tts_pool.sample_rate
        n = int(sr * gap_ms / 1000)
        try:
            import numpy as np

            data = np.zeros(n, dtype=np.int16).tobytes()
            return rtc.AudioFrame(
                data=data, sample_rate=sr, num_channels=1, samples_per_channel=n
            )
        except Exception:
            return None

    # ── Изолированные вызовы LiveKit SDK (правятся при апдейте SDK) ───────────
    def _lk_register_interruption_handler(self) -> None:
        """
        Регистрирует реакцию на прерывание. Основной надёжный механизм — это
        CancelledError в node-генераторах (обрабатывается там). Здесь дополнительно
        подписываемся на событие сессии для явной 5-шаговой оркестрации/логирования.
        Имя события уточняется по версии SDK.
        """
        session = self._session
        if session is None:
            return

        def _on_interrupt(*_args, **_kwargs) -> None:
            asyncio.create_task(self._handle_barge_in(reason="framework_interruption"))

        for event_name in ("agent_speech_interrupted", "user_started_speaking", "interruption"):
            with contextlib.suppress(Exception):
                session.on(event_name, _on_interrupt)

    async def _lk_stop_agent_playout(self) -> None:
        """Шаг 1: остановить проигрывание речи агента (идемпотентно)."""
        session = self._session
        if session is None:
            return
        for method in ("interrupt", "stop_speech", "clear_playout"):
            fn = getattr(session, method, None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                    return

    async def _lk_flush_output_buffers(self) -> None:
        """Шаг 3: очистить исходящий аудио-буфер (SFU output)."""
        session = self._session
        if session is None:
            return
        out = getattr(session, "output", None)
        for target in (out, session):
            for method in ("flush", "clear", "clear_buffer"):
                fn = getattr(target, method, None) if target is not None else None
                if callable(fn):
                    with contextlib.suppress(Exception):
                        res = fn()
                        if asyncio.iscoroutine(res):
                            await res
                        return

    async def _lk_reset_endpoint(self) -> None:
        """Шаг 5: сбросить endpoint-состояние ASR, начать новый турн."""
        session = self._session
        if session is None:
            return
        for method in ("commit_user_turn", "clear_user_turn", "reset_turn"):
            fn = getattr(session, method, None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                    return

    async def _lk_send_subtitle(self, text: str) -> None:
        """Text-only деградация: отправить перевод как data-сообщение на клиент."""
        session = self._session
        if session is None:
            return
        with contextlib.suppress(Exception):
            room = getattr(getattr(session, "job_ctx", None), "room", None)
            if room is not None and hasattr(room, "local_participant"):
                await room.local_participant.publish_data(
                    payload=text.encode("utf-8"), topic="synctona.subtitle"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Прогрев и admission control
# ─────────────────────────────────────────────────────────────────────────────
class AdmissionController:
    """§2.2(a) ТЗ: семафор по глобальной ёмкости TTS + гейт по утилизации пула."""

    def __init__(self, settings: Settings, tts: TTSClient) -> None:
        self._s = settings
        self._tts = tts
        self._sem = asyncio.Semaphore(settings.global_max_concurrent_tts)

    async def try_admit(self) -> bool:
        # 1) есть ли свободный слот в глобальном лимите
        if self._sem.locked() or self._sem._value <= 0:  # быстрая проверка без блокировки
            logger.warning("Admission отказ: глобальный лимит TTS исчерпан")
            return False
        # 2) не перегрет ли пул прямо сейчас
        try:
            util = await self._tts.utilization()
        except Exception:
            util = 0.0
        if util >= self._s.admission.gpu_util_admit_threshold:
            logger.warning("Admission отказ: утилизация пула %.2f ≥ порога", util)
            return False
        await self._sem.acquire()
        return True

    def release(self) -> None:
        with contextlib.suppress(ValueError):
            self._sem.release()


async def warmup_models(settings: Settings, nmt: NMTClient, tts: TTSClient) -> None:
    """
    Прогрев с ретраями. Если модели не прогрелись — сессии принимать нельзя,
    поэтому исключение пробрасывается наверх (worker не встанет в READY).
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, settings.runtime.warmup_retries + 1):
        try:
            await asyncio.gather(nmt.warmup(), tts.warmup())
            logger.info("Прогрев моделей завершён (попытка %d)", attempt)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("Прогрев не удался (попытка %d/%d): %s",
                           attempt, settings.runtime.warmup_retries, e)
            await asyncio.sleep(settings.runtime.warmup_backoff_s * attempt)
    raise RuntimeError(f"Прогрев моделей провалился: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа worker'а
# ─────────────────────────────────────────────────────────────────────────────
server = AgentServer()


def _build_clients(settings: Settings) -> tuple[NMTClient, TTSClient]:
    """
    Фабрика клиентов. Реальные реализации подключаются на Step 2/3:
        from synctona.clients.nmt_client import HttpNMTClient
        from synctona.clients.tts_client import BatchedTTSClient
    До этого момента worker поднимается, но откажет при первой сессии на warmup —
    это осознанно (fail-fast), чтобы не отдавать пустой звук.
    """
    from synctona.clients.nmt_client import HttpNMTClient  # noqa: WPS433 (Step 3)
    from synctona.clients.tts_client import BatchedTTSClient  # noqa: WPS433 (Step 2)

    nmt = HttpNMTClient(settings.nmt)
    tts = BatchedTTSClient(settings.tts_pool)
    return nmt, tts


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.observability.log_level)

    nmt, tts = _build_clients(settings)
    admission = AdmissionController(settings, tts)

    # Прогрев (разово на процесс уместнее в prewarm-хуке worker'а; здесь — guard).
    try:
        await warmup_models(settings, nmt, tts)
    except Exception:
        logger.exception("Модели не прогрелись — сессия отклонена (session=%s)", ctx.room.name)
        await ctx.shutdown(reason="warmup_failed")
        return

    # Admission control до подключения к комнате.
    if not await admission.try_admit():
        metrics.ADMISSION_REJECTS.inc()
        with contextlib.suppress(Exception):
            await ctx.room.local_participant.publish_data(
                payload=settings.admission.reject_message.encode("utf-8"),
                topic="synctona.system",
            )
        await ctx.shutdown(reason="overloaded")
        return
    
    # Успешный вход
    metrics.ADMISSION_ADMITS.inc()
    metrics.ACTIVE_SESSIONS.inc()

    registry = RedisSpeakerRegistry(os.getenv("SYNCTONA_REDIS_URL", "redis://redis:6379/0"))
    runtime = SessionRuntime(
        session_id=ctx.room.name,
        src_lang=settings.runtime.default_source_lang,
        tgt_lang=settings.runtime.default_target_lang,
        speaker_voice=await _resolve_speaker_voice(ctx, registry, settings),
    )

    # Разрыв сети / отключение участника → зачистка и освобождение слота admission.
    def _on_disconnected(*_a, **_kw) -> None:
        logger.info("Отключение (session=%s) — зачистка", runtime.session_id)
        metrics.ACTIVE_SESSIONS.dec()  # Уменьшаем счётчик при штатном дисконнекте
        admission.release()

    for ev in ("disconnected", "participant_disconnected"):
        with contextlib.suppress(Exception):
            ctx.room.on(ev, _on_disconnected)

    agent = SynctonaTranslatorAgent(settings=settings, nmt=nmt, tts=tts, runtime=runtime)

    session = AgentSession(
        stt=SynctonaSTT(settings, language=runtime.src_lang),
        vad=None,
        allow_interruptions=True,
        min_endpointing_delay=settings.endpointing.min_endpointing_delay_s,
        max_endpointing_delay=settings.endpointing.max_endpointing_delay_s,
        aec_warmup_duration=settings.livekit.aec_warmup_s,
    )

    try:
        await session.start(agent=agent, room=ctx.room)
        await ctx.wait_for_shutdown()
    except asyncio.CancelledError:
        logger.info("Worker отменён (session=%s)", runtime.session_id)
        raise
    except Exception:
        logger.exception("Фатальная ошибка сессии (session=%s)", runtime.session_id)
    finally:
        with contextlib.suppress(Exception):
            await agent._cleanup_turn_resources(reason="finally")

        # Гарантированный декремент сессии при любом исходе
        metrics.ACTIVE_SESSIONS.dec()
        admission.release()
        with contextlib.suppress(Exception):
            await session.aclose()


async def _resolve_speaker_voice(ctx, registry, settings) -> str:
    for p in ctx.room.remote_participants.values():
        voice = await registry.resolve_voice(p.identity)   # None, если нет consent
        if voice:
            return voice
    return settings.tts_pool.generic_voice_id


def main() -> None:
    metrics.start_metrics_server(int(os.getenv("SYNCTONA_METRICS_PORT", "9100")))
    tracing.setup_tracing("synctona-agent", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
    cli.run_app(server)


if __name__ == "__main__":
    main()