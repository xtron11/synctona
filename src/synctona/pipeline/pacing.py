"""
Synctona — pacing и склейка аудио (Step 2, §4 ТЗ).

Мелкие DSP-хелперы для ровного воспроизведения:
  • silence_frame — микропауза на стыке предложений (rtc.AudioFrame тишины);
  • recommend_jitter_ms — рекомендация размера клиентского буфера под наблюдаемый джиттер;
  • crossfade_pcm16 — сглаживание стыка двух PCM16-кусков (убирает щелчки).
"""

from __future__ import annotations

from typing import Optional


def silence_pcm16(sample_rate: int, gap_ms: int) -> bytes:
    if gap_ms <= 0:
        return b""
    n = int(sample_rate * gap_ms / 1000)
    return b"\x00\x00" * n  # int16 нули


def silence_frame(sample_rate: int, gap_ms: int):
    """rtc.AudioFrame тишины на gap_ms (или None, если недоступен rtc/gap<=0)."""
    if gap_ms <= 0:
        return None
    try:
        from livekit import rtc

        data = silence_pcm16(sample_rate, gap_ms)
        n = len(data) // 2
        return rtc.AudioFrame(
            data=data, sample_rate=sample_rate, num_channels=1, samples_per_channel=n
        )
    except Exception:
        return None


def recommend_jitter_ms(observed_jitter_ms: float, floor_ms: int = 150, ceil_ms: int = 250) -> int:
    """
    Простое правило: буфер ≈ 2× наблюдаемого джиттера, зажатый в [floor, ceil].
    Держит баланс между устойчивостью к потерям и добавленной задержкой (§4 ТЗ).
    """
    target = int(observed_jitter_ms * 2)
    return max(floor_ms, min(ceil_ms, target))


def crossfade_pcm16(prev_tail: bytes, next_head: bytes, fade_samples: int = 240) -> bytes:
    """
    Линейный кроссфейд на стыке двух PCM16-моно кусков, чтобы убрать щелчок.
    Возвращает микшированный участок длиной min(fade_samples, len). Остальное
    вызывающий склеивает как есть.
    """
    try:
        import numpy as np
    except Exception:
        return next_head

    a = np.frombuffer(prev_tail, dtype=np.int16).astype(np.float32)
    b = np.frombuffer(next_head, dtype=np.int16).astype(np.float32)
    n = min(fade_samples, len(a), len(b))
    if n == 0:
        return next_head
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    mixed = a[-n:] * fade_out + b[:n] * fade_in
    mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
    return mixed.tobytes() + b[n:].astype(np.int16).tobytes()