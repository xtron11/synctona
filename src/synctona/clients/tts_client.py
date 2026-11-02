"""
Synctona — клиент к TTS-пулу (Step 2).

Реализует контракты из Step 1 (agent_worker.py):
    TTSClient   — warmup(), utilization(), synthesize(...) -> TTSStream
    TTSStream   — .handle + async-итерация rtc.AudioFrame
    TTSJobHandle — .job_id + cancel()

synthesize() возвращает объект-стрим СИНХРОННО, чтобы handle был доступен сразу
(его регистрирует tts_node для отмены при barge-in). WS открывается лениво на
первой итерации.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator, Optional

import aiohttp
from livekit import rtc

from synctona.config import TTSPoolSettings

logger = logging.getLogger("synctona.tts.client")


class _RemoteJobHandle:
    def __init__(self, client: "BatchedTTSClient", job_id: str) -> None:
        self._client = client
        self.job_id = job_id

    async def cancel(self) -> None:
        """Шаг 2 barge-in: сигнал пулу прекратить генерацию и освободить слот GPU."""
        url = f"{self._client.base_url}/v1/cancel/{self.job_id}"
        try:
            async with self._client.session.post(url, timeout=aiohttp.ClientTimeout(total=2)) as r:
                await r.read()
        except Exception as e:  # noqa: BLE001
            logger.warning("Не удалось отменить джобу %s: %s", self.job_id, e)


class _RemoteTTSStream:
    """Ленивый WS-стрим: PCM16 с пула → rtc.AudioFrame."""

    def __init__(
        self,
        client: "BatchedTTSClient",
        *,
        text: str,
        session_id: str,
        voice: str,
        preset: str,
        sample_rate: int,
    ) -> None:
        self._client = client
        self._text = text
        self._session_id = session_id
        self._voice = voice
        self._preset = preset
        self._sr = sample_rate
        self.job_id = f"{session_id}:{uuid.uuid4().hex[:12]}"
        self.handle = _RemoteJobHandle(client, self.job_id)

    async def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        ws_url = f"{self._client.ws_url}/v1/synthesize"
        try:
            async with self._client.session.ws_connect(ws_url, heartbeat=15) as ws:
                await ws.send_json(
                    {
                        "job_id": self.job_id,
                        "text": self._text,
                        "voice_id": self._voice,
                        "preset": self._preset,
                        "sample_rate": self._sr,
                    }
                )
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        frame = self._pcm_to_frame(msg.data)
                        if frame is not None:
                            yield frame
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        event = json.loads(msg.data).get("event")
                        if event in ("eof", "error", "rejected"):
                            if event != "eof":
                                logger.warning("TTS-пул завершил стрим событием '%s' (job=%s)",
                                               event, self.job_id)
                            return
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return
        except Exception:
            logger.exception("Ошибка WS-стрима TTS (job=%s)", self.job_id)
            raise

    def _pcm_to_frame(self, data: bytes) -> Optional[rtc.AudioFrame]:
        if not data:
            return None
        samples = len(data) // 2  # int16
        if samples == 0:
            return None
        return rtc.AudioFrame(
            data=data, sample_rate=self._sr, num_channels=1, samples_per_channel=samples
        )


class BatchedTTSClient:
    """Клиент к пулу; один общий aiohttp-сессионный пул на процесс-worker."""

    def __init__(self, settings: TTSPoolSettings) -> None:
        self._s = settings
        self.base_url = settings.endpoint.rstrip("/")
        # ws:// из http://
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def warmup(self) -> None:
        url = f"{self.base_url}/v1/warmup"
        async with self.session.post(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
            body = await r.json()
            if not body.get("ok"):
                raise RuntimeError(f"Прогрев TTS-пула не удался: {body}")
        logger.info("TTS-пул прогрет: %s", self.base_url)

    async def utilization(self) -> float:
        url = f"{self.base_url}/v1/utilization"
        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as r:
            body = await r.json()
            return float(body.get("utilization", 0.0))

    def synthesize(
        self,
        text: str,
        *,
        session_id: str,
        voice: str,
        preset: str,
        sample_rate: int,
    ) -> _RemoteTTSStream:
        return _RemoteTTSStream(
            self,
            text=text,
            session_id=session_id,
            voice=voice,
            preset=preset,
            sample_rate=sample_rate,
        )

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()