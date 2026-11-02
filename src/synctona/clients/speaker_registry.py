"""
Synctona — реестр голосов и consent (Step 3).

Redis-хранилище связки «участник → voice_id» с гейтом по consent. Клонирование
голоса без явного согласия запрещено (§3/§7 ТЗ: deepfake/регуляторный риск),
поэтому resolve_voice() возвращает voice_id ТОЛЬКО при наличии consent.

Ключи:
  synctona:voice:{identity}    → voice_id (str)
  synctona:consent:{identity}  → "1" | "0"

Реализует Protocol SpeakerRegistry из base.py.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("synctona.speaker")

_VOICE_KEY = "synctona:voice:{}"
_CONSENT_KEY = "synctona:consent:{}"


class RedisSpeakerRegistry:
    def __init__(self, redis_url: str = "redis://redis:6379/0", *, ttl_s: int = 86_400) -> None:
        self._url = redis_url
        self._ttl = ttl_s
        self._redis = None

    async def _client(self):
        if self._redis is None:
            import redis.asyncio as redis  # type: ignore

            self._redis = redis.from_url(self._url, decode_responses=True)
        return self._redis

    async def has_consent(self, identity: str) -> bool:
        try:
            r = await self._client()
            val = await r.get(_CONSENT_KEY.format(identity))
            return val == "1"
        except Exception:
            logger.exception("Redis consent lookup failed для %s", identity)
            return False

    async def resolve_voice(self, identity: str) -> Optional[str]:
        """voice_id при наличии consent, иначе None (→ generic-голос)."""
        if not await self.has_consent(identity):
            logger.info("Нет consent на клонирование для %s → generic", identity)
            return None
        try:
            r = await self._client()
            return await r.get(_VOICE_KEY.format(identity))
        except Exception:
            logger.exception("Redis voice lookup failed для %s", identity)
            return None

    async def register_voice(self, identity: str, voice_id: str, *, consent: bool) -> None:
        """Сохраняет voice_id и фиксирует consent (вызывается из consent-flow клиента)."""
        r = await self._client()
        pipe = r.pipeline()
        pipe.set(_VOICE_KEY.format(identity), voice_id, ex=self._ttl)
        pipe.set(_CONSENT_KEY.format(identity), "1" if consent else "0", ex=self._ttl)
        await pipe.execute()
        logger.info("Голос зарегистрирован: %s → %s (consent=%s)", identity, voice_id, consent)

    async def revoke(self, identity: str) -> None:
        r = await self._client()
        pipe = r.pipeline()
        pipe.delete(_VOICE_KEY.format(identity))
        pipe.set(_CONSENT_KEY.format(identity), "0", ex=self._ttl)
        await pipe.execute()

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()