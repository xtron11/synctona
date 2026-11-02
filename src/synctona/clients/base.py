"""
Synctona — контракты внешних сервисов (Step 3).

Единый источник Protocol-интерфейсов ASR/NMT/TTS. В Step 1 они были объявлены
внутри agent_worker.py; теперь живут здесь. agent_worker может импортировать их
отсюда:

    from synctona.clients.base import NMTClient, TTSClient, TTSStream, TTSJobHandle

Protocol'ы структурные (runtime_checkable), поэтому конкретные клиенты
(HttpNMTClient, BatchedTTSClient, SynctonaSTT) их НЕ наследуют — достаточно
совпадения сигнатур. Это сохраняет слабую связанность.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterable, Optional, Protocol, runtime_checkable

from livekit import rtc


# ─────────────────────────── NMT ────────────────────────────
@runtime_checkable
class NMTClient(Protocol):
    async def warmup(self) -> None: ...

    async def translate(
        self, text: str, src: str, tgt: str, *, beam_size: int, context: list[str]
    ) -> str:
        """Переводит ОДНО предложение целиком (§4 ТЗ)."""
        ...


# ─────────────────────────── TTS ────────────────────────────
@runtime_checkable
class TTSJobHandle(Protocol):
    job_id: str

    async def cancel(self) -> None: ...


@runtime_checkable
class TTSStream(Protocol):
    handle: TTSJobHandle

    def __aiter__(self) -> AsyncIterable[rtc.AudioFrame]: ...


@runtime_checkable
class TTSClient(Protocol):
    async def warmup(self) -> None: ...

    async def utilization(self) -> float: ...

    def synthesize(
        self,
        text: str,
        *,
        session_id: str,
        voice: str,
        preset: str,
        sample_rate: int,
    ) -> TTSStream: ...


# ─────────────────────────── ASR ────────────────────────────
class TranscriptType(Enum):
    PARTIAL = auto()      # нестабильный промежуточный результат
    FINAL = auto()        # финализированное предложение
    ENDPOINT = auto()     # обнаружен конец речи (можно закрывать турн)


@dataclass
class TranscriptEvent:
    type: TranscriptType
    text: str
    language: str = ""
    is_sentence_end: bool = False


@runtime_checkable
class SpeakerRegistry(Protocol):
    async def resolve_voice(self, identity: str) -> Optional[str]:
        """voice_id для участника, если есть consent; иначе None."""
        ...

    async def has_consent(self, identity: str) -> bool: ...