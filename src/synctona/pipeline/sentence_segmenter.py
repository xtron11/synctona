"""
Synctona — сегментация и нормализация текста для TTS (Step 2, §4 ТЗ).

Заменяет regex-заглушку из agent_worker.py на:
  • сегментацию по предложениям (pysbd — устойчив к сокращениям/числам, с regex-фолбэком);
  • нормализацию перед TTS: числа → слова, аббревиатуры → произносимая форма,
    чистка символов, на которых TTS спотыкается (растёт TTFB).

Единица перевода/озвучки — предложение (§4). Guard длинных фраз — в chunker.py.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("synctona.pipeline.segmenter")

# Аббревиатуры, которые нельзя принять за конец предложения (EN + общие).
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "inc", "ltd", "co", "no", "fig", "al", "u.s", "u.k",
}

# Разворачиваемые аббревиатуры при произношении (пример; расширяется по домену).
_SPOKEN = {
    "AI": "A I",
    "API": "A P I",
    "GPU": "G P U",
    "TTS": "text to speech",
    "STT": "speech to text",
    "ID": "I D",
}


class Segmenter:
    def __init__(self, language: str = "en") -> None:
        self._lang = language
        self._sbd = self._make_sbd(language)

    @staticmethod
    def _make_sbd(language: str):
        try:
            import pysbd  # type: ignore

            return pysbd.Segmenter(language=language, clean=False)
        except Exception:
            logger.warning("pysbd недоступен — использую regex-фолбэк сегментации.")
            return None

    # ── Сегментация ──────────────────────────────────────────────────────────
    def segment(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self._sbd is not None:
            try:
                return [s.strip() for s in self._sbd.segment(text) if s.strip()]
            except Exception:
                logger.exception("pysbd упал — фолбэк на regex.")
        return self._regex_segment(text)

    @staticmethod
    def _regex_segment(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?…])\s+", text)
        out: list[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Склеиваем ложные разрывы после аббревиатур.
            if out:
                last_word = out[-1].split()[-1].rstrip(".").lower() if out[-1].split() else ""
                if last_word in _ABBREVIATIONS:
                    out[-1] = f"{out[-1]} {p}"
                    continue
            out.append(p)
        return out

    # ── Нормализация под TTS ─────────────────────────────────────────────────
    def normalize_for_tts(self, text: str) -> str:
        text = self._expand_abbreviations(text)
        text = self._expand_numbers(text)
        text = self._clean_symbols(text)
        return text.strip()

    @staticmethod
    def _expand_abbreviations(text: str) -> str:
        def repl(m: re.Match) -> str:
            token = m.group(0)
            return _SPOKEN.get(token, token)

        return re.sub(r"\b[A-Z]{2,}\b", repl, text)

    def _expand_numbers(self, text: str) -> str:
        try:
            from num2words import num2words  # type: ignore
        except Exception:
            return text

        def repl(m: re.Match) -> str:
            raw = m.group(0)
            try:
                if "." in raw:
                    whole, frac = raw.split(".", 1)
                    words = num2words(int(whole), lang=self._lang)
                    digits = " ".join(num2words(int(d), lang=self._lang) for d in frac)
                    return f"{words} point {digits}"
                return num2words(int(raw), lang=self._lang)
            except Exception:
                return raw

        return re.sub(r"\d+(?:\.\d+)?", repl, text)

    @staticmethod
    def _clean_symbols(text: str) -> str:
        text = text.replace("&", " and ").replace("%", " percent ").replace("@", " at ")
        text = re.sub(r"[*_`#>|]", " ", text)          # markdown-мусор
        text = re.sub(r"\s{2,}", " ", text)
        return text