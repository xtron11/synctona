"""
Synctona — интеграционный тест каскада (Step 3).

Проверяет сквозную ЛОГИКУ RU→EN без GPU/LiveKit: реальные Segmenter + SentenceChunker
(Step 2) + фейковые NMT/TTS, реализующие контракты из base.py. Мимикрирует то, что
делают llm_node/tts_node оркестратора: segment → перевод по предложению → chunk →
стриминговый синтез, с пайплайнингом и barge-in.

Запуск:  pytest tests/test_integration.py -v
(в собранном репозитории с установленным пакетом synctona)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from synctona.pipeline.sentence_segmenter import Segmenter
from synctona.pipeline.chunker import SentenceChunker


# ─────────────────────────── Фейки под контракты ────────────────────────────
@dataclass
class FakeFrame:
    data: bytes


@dataclass
class FakeJobHandle:
    job_id: str
    cancelled: bool = False

    async def cancel(self) -> None:
        self.cancelled = True


class FakeTTSStream:
    """Стрим, отдающий N чанков; уважает отмену через handle (как реальный)."""

    def __init__(self, job_id: str, text: str, chunks: int = 3, delay: float = 0.0) -> None:
        self.handle = FakeJobHandle(job_id=job_id)
        self._text = text
        self._chunks = chunks
        self._delay = delay

    async def __aiter__(self):
        for i in range(self._chunks):
            if self.handle.cancelled:
                return
            if self._delay:
                await asyncio.sleep(self._delay)
            yield FakeFrame(data=f"{self._text}#{i}".encode())


class FakeTTSClient:
    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[str] = []
        self._util = 0.1
        self._delay = delay
        self.last_stream: FakeTTSStream | None = None

    async def warmup(self) -> None:
        pass

    async def utilization(self) -> float:
        return self._util

    def synthesize(self, text, *, session_id, voice, preset, sample_rate) -> FakeTTSStream:
        self.calls.append(text)
        self.last_stream = FakeTTSStream(f"{session_id}:{len(self.calls)}", text, delay=self._delay)
        return self.last_stream


class FakeNMTClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def warmup(self) -> None:
        pass

    async def translate(self, text, src, tgt, *, beam_size, context) -> str:
        self.calls.append(text)
        await asyncio.sleep(0)  # имитируем сетевой хоп
        return f"[{tgt}]{text}"


# ─────────────────────────── Мини-оркестратор каскада ───────────────────────
async def run_cascade(
    source_text: str,
    nmt: FakeNMTClient,
    tts: FakeTTSClient,
    *,
    segmenter: Segmenter,
    chunker: SentenceChunker,
    src: str = "ru",
    tgt: str = "en",
    cancel_after_frames: int | None = None,
) -> list[FakeFrame]:
    """
    Повторяет логику llm_node→tts_node: предложение переводится и тут же уходит в
    TTS, пока переводится следующее (пайплайнинг). cancel_after_frames имитирует
    barge-in: после N кадров дёргаем handle.cancel().
    """
    frames: list[FakeFrame] = []
    produced = 0

    for sentence in segmenter.segment(source_text):
        translated = await nmt.translate(sentence, src, tgt, beam_size=4, context=[])
        for chunk in chunker.chunk_sentence(translated):
            stream = tts.synthesize(
                chunk.text, session_id="s1", voice="v", preset="default", sample_rate=24000
            )
            async for frame in stream.__aiter__():
                frames.append(frame)
                produced += 1
                if cancel_after_frames is not None and produced >= cancel_after_frames:
                    await stream.handle.cancel()  # шаг 2 barge-in
                    return frames
    return frames


# ─────────────────────────── Фикстуры ───────────────────────────────────────
@pytest.fixture
def segmenter() -> Segmenter:
    return Segmenter(language="en")


@pytest.fixture
def chunker(segmenter: Segmenter) -> SentenceChunker:
    return SentenceChunker(segmenter, max_sentence_words=14, clause_split_chars=",;:—")


# ─────────────────────────── Тесты ──────────────────────────────────────────
@pytest.mark.asyncio
async def test_cascade_order_and_completeness(segmenter, chunker):
    nmt, tts = FakeNMTClient(), FakeTTSClient()
    text = "Первое предложение. Второе предложение! Третье?"
    frames = await run_cascade(text, nmt, tts, segmenter=segmenter, chunker=chunker)

    assert len(nmt.calls) == 3
    assert len(tts.calls) >= 3
    assert frames and all(isinstance(f, FakeFrame) for f in frames)


@pytest.mark.asyncio
async def test_pipelining_translations_incremental(segmenter, chunker):
    """NMT-вызовы идут по одному предложению (а не одним куском) — это пайплайнинг."""
    nmt, tts = FakeNMTClient(), FakeTTSClient()
    text = "Alpha one. Beta two. Gamma three."
    await run_cascade(text, nmt, tts, segmenter=segmenter, chunker=chunker, src="en", tgt="ru")
    assert nmt.calls == ["Alpha one.", "Beta two.", "Gamma three."]


@pytest.mark.asyncio
async def test_long_sentence_clause_guard(segmenter, chunker):
    """Длинное предложение режется на несколько TTS-чанков (guard TTFB)."""
    nmt, tts = FakeNMTClient(), FakeTTSClient()
    long_sentence = (
        "This is a very long sentence, with several clauses, that keeps going on, "
        "and on, and should be split into multiple chunks."
    )
    await run_cascade(long_sentence, nmt, tts, segmenter=segmenter, chunker=chunker, src="en", tgt="ru")
    assert len(tts.calls) > 1


@pytest.mark.asyncio
async def test_barge_in_cancels_tts(segmenter, chunker):
    """Barge-in: после отмены handle помечен cancelled и стрим прекращается."""
    nmt, tts = FakeNMTClient(), FakeTTSClient()
    text = "One long enough sentence to stream several chunks here."
    frames = await run_cascade(
        text, nmt, tts, segmenter=segmenter, chunker=chunker,
        src="en", tgt="ru", cancel_after_frames=1,
    )
    assert tts.last_stream is not None
    assert tts.last_stream.handle.cancelled is True   # шаг 2 barge-in сработал
    assert len(frames) == 1                           # после отмены кадры не идут


@pytest.mark.asyncio
async def test_nmt_translation_shape(segmenter, chunker):
    nmt, tts = FakeNMTClient(), FakeTTSClient()
    await run_cascade("Привет мир.", nmt, tts, segmenter=segmenter, chunker=chunker)
    assert any(c.startswith("[en]") for c in tts.calls)