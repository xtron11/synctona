"""
Synctona — нарезка предложений для TTS (Step 2, §4 ТЗ).

Из потока переведённых предложений формирует TTS-готовые чанки, соблюдая баланс
контекст↔TTFB:
  • единица — целое предложение (порядок слов RU→EN и т.п. ломать нельзя);
  • аномально длинные предложения режем ПО КЛАУЗАМ (запятые/союзы/двоеточия),
    чтобы ограничить TTFB, жертвуя малой прозодической склейкой;
  • нормализация текста делегируется Segmenter.normalize_for_tts.

Порядок применения на выходе NMT:  translated_sentence → normalize → chunk → TTS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterable, AsyncIterator, Iterable

from synctona.pipeline.sentence_segmenter import Segmenter


@dataclass
class TTSChunk:
    text: str
    is_sentence_end: bool   # True на последнем куске предложения → пора вставить join-gap


class SentenceChunker:
    def __init__(
        self,
        segmenter: Segmenter,
        *,
        max_sentence_words: int,
        clause_split_chars: str,
        min_clause_words: int = 3,
    ) -> None:
        self._seg = segmenter
        self._max_words = max_sentence_words
        self._clause_chars = set(clause_split_chars)
        self._min_clause_words = min_clause_words

    # ── Guard длинного предложения → список клауз ─────────────────────────────
    def split_sentence(self, sentence: str) -> list[str]:
        words = sentence.split()
        if len(words) <= self._max_words:
            return [sentence]

        clauses: list[str] = []
        buf: list[str] = []
        for token in words:
            buf.append(token)
            ends_clause = bool(token) and token[-1] in self._clause_chars
            if ends_clause and len(buf) >= self._min_clause_words:
                clauses.append(" ".join(buf))
                buf = []
        if buf:
            # Если хвост слишком короткий — приклеиваем к предыдущей клаузе.
            if clauses and len(buf) < self._min_clause_words:
                clauses[-1] = f"{clauses[-1]} {' '.join(buf)}"
            else:
                clauses.append(" ".join(buf))
        return clauses or [sentence]

    # ── Синхронный путь: одно переведённое предложение → чанки ────────────────
    def chunk_sentence(self, sentence: str) -> list[TTSChunk]:
        normalized = self._seg.normalize_for_tts(sentence)
        pieces = self.split_sentence(normalized)
        chunks: list[TTSChunk] = []
        for idx, piece in enumerate(pieces):
            chunks.append(TTSChunk(text=piece, is_sentence_end=(idx == len(pieces) - 1)))
        return chunks

    def chunk_many(self, sentences: Iterable[str]) -> list[TTSChunk]:
        out: list[TTSChunk] = []
        for s in sentences:
            out.extend(self.chunk_sentence(s))
        return out

    # ── Асинхронный путь: поток предложений от NMT → поток чанков ─────────────
    async def chunk_stream(
        self, sentences: AsyncIterable[str]
    ) -> AsyncIterator[TTSChunk]:
        async for sentence in sentences:
            if not sentence.strip():
                continue
            for chunk in self.chunk_sentence(sentence):
                yield chunk