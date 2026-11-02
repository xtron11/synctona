"""
Synctona — ASR-адаптер как LiveKit STT-плагин (Step 3).

Мост между AgentSession и нашим стриминговым ASR-сервисом. Реализует
livekit.agents.stt.STT со стримингом: аудио-фреймы сессии уходят по WS в ASR-пул,
а партиалы/финалы возвращаются как stt.SpeechEvent (INTERIM/FINAL/END_OF_SPEECH),
которые фреймворк использует для turn-detection и запуска NMT-ноды.

ВНИМАНИЕ по версии SDK: имена enum'ов SpeechEventType и способ отдачи событий в
livekit-agents эволюционируют. Всё, что зависит от SDK, изолировано в _emit()/
capabilities и помечено; при апдейте правится только там.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import aiohttp
from livekit import rtc
from livekit.agents import stt as lk_stt
from livekit.agents import utils as lk_utils

from synctona.config import Settings

logger = logging.getLogger("synctona.asr.adapter")


class SynctonaSTT(lk_stt.STT):
    """STT-плагин, проксирующий на self-hosted ASR-пул."""

    def __init__(self, settings: Settings, *, language: str) -> None:
        super().__init__(
            capabilities=lk_stt.STTCapabilities(streaming=True, interim_results=True)
        )
        self._s = settings
        self._language = language
        self.base_url = settings.nmt.endpoint  # переопределяется ниже реальным ASR endpoint
        self._asr_http = _asr_base_url(settings)
        self._ws_url = self._asr_http.replace("http://", "ws://").replace("https://", "wss://")
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):  # noqa: ANN001
        # Синхронное распознавание не используем (пайплайн стриминговый), но контракт
        # требует метод. Возвращаем пустой результат.
        return lk_stt.SpeechEvent(
            type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[lk_stt.SpeechData(text="", language=self._language)],
        )

    def stream(self, *, language=None, conn_options=None) -> "SynctonaSTTStream":  # noqa: ANN001
        return SynctonaSTTStream(self, language or self._language)

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


class SynctonaSTTStream(lk_stt.SpeechStream):
    """Один стрим: качает фреймы сессии в ASR-пул и эмитит SpeechEvent."""

    def __init__(self, parent: SynctonaSTT, language: str) -> None:
        super().__init__(stt=parent, conn_options=lk_utils.NOT_GIVEN, sample_rate=16000)
        self._parent = parent
        self._language = language

    async def _run(self) -> None:
        """
        Читает фреймы из self._input_ch, шлёт в ASR по WS, события кладёт в
        self._event_ch. Имена каналов соответствуют базовому SpeechStream 1.x.
        """
        ws_url = f"{self._parent._ws_url}/v1/stream"
        try:
            async with self._parent._http.ws_connect(ws_url, heartbeat=15) as ws:
                await ws.send_json(
                    {"session_id": id(self), "language": self._language, "sample_rate": 16000}
                )
                sender = asyncio.create_task(self._pump_audio(ws))
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._emit(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                finally:
                    sender.cancel()
        except Exception:
            logger.exception("Ошибка ASR-стрима (adapter)")
            raise

    async def _pump_audio(self, ws) -> None:
        try:
            async for frame in self._input_ch:
                if isinstance(frame, rtc.AudioFrame):
                    await ws.send_bytes(bytes(frame.data))
            # Вход закрылся → просим ASR доотдать финал.
            await ws.send_str(json.dumps({"event": "finalize"}))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Сбой отправки аудио в ASR")

    def _emit(self, event: dict) -> None:
        """
        Маппинг событий ASR-сервиса → stt.SpeechEvent. Изолированная точка SDK.
        """
        etype = event.get("type")
        text = event.get("text", "")
        if etype == "partial":
            self._event_ch.send_nowait(
                lk_stt.SpeechEvent(
                    type=lk_stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[lk_stt.SpeechData(text=text, language=self._language)],
                )
            )
        elif etype == "final":
            self._event_ch.send_nowait(
                lk_stt.SpeechEvent(
                    type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[lk_stt.SpeechData(text=text, language=self._language)],
                )
            )
        elif etype == "endpoint":
            self._event_ch.send_nowait(
                lk_stt.SpeechEvent(type=lk_stt.SpeechEventType.END_OF_SPEECH)
            )


def _asr_base_url(settings: Settings) -> str:
    """
    ASR endpoint. В config добавляется отдельная секция; пока берём из окружения
    с дефолтом, чтобы не ломать config из Step 1.
    """
    import os

    return os.getenv("SYNCTONA_ASR_ENDPOINT", "http://asr:8003").rstrip("/")