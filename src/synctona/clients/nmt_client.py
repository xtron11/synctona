"""
Synctona — клиент к NMT-сервису (Step 3).

Реализует контракт NMTClient из Step 1/base.py: warmup() + translate() по одному
предложению. Rolling-контекст принимается в сигнатуре (для совместимости), но
NMT-сервис на M2M его пока не использует — см. nmt_server/engine.py.
"""

from __future__ import annotations

import logging

import aiohttp

from synctona.config import NMTSettings

logger = logging.getLogger("synctona.nmt.client")


class HttpNMTClient:
    def __init__(self, settings: NMTSettings) -> None:
        self._s = settings
        self.base_url = settings.endpoint.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

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
                raise RuntimeError(f"Прогрев NMT не удался: {body}")
        logger.info("NMT-сервис прогрет: %s", self.base_url)

    async def translate(
        self, text: str, src: str, tgt: str, *, beam_size: int, context: list[str]
    ) -> str:
        if not text.strip():
            return ""
        url = f"{self.base_url}/v1/translate"
        payload = {"text": text, "src": src, "tgt": tgt, "beam": beam_size}
        timeout = aiohttp.ClientTimeout(total=self._s.request_timeout_s)
        async with self.session.post(url, json=payload, timeout=timeout) as r:
            if r.status != 200:
                detail = await r.text()
                raise RuntimeError(f"NMT {r.status}: {detail}")
            body = await r.json()
            return body.get("translation", "")

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()