"""
Synctona NMT-сервис — движок (Step 3).

Мультиязычный перевод на M2M-100 (по умолчанию 418M; NLLB подключается тем же
интерфейсом сменой имени модели и кодов языков). NMT дёшев и хорошо батчируется,
поэтому здесь настоящий dynamic micro-batching:
  • запросы копятся окном max_wait_ms или до max_batch;
  • группируются по паре (src, tgt), т.к. forced_bos_token_id зависит от target;
  • батч токенизируется и генерируется одним прогоном.

Перевод — на уровне предложения (§4). Rolling-контекст принимается, но по умолчанию
не инжектится в модель (M2M — sentence-level MT); хук оставлен для context-aware
модели. Greedy при деградации приходит из beam_size=1 (ставит агент).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("synctona.nmt.engine")


@dataclass
class _PendingReq:
    text: str
    src: str
    tgt: str
    beam: int
    future: asyncio.Future = field(default_factory=asyncio.Future)


class M2MEngine:
    def __init__(
        self,
        *,
        model_name: str = "facebook/m2m100_418M",
        device: str = "cuda",
        max_batch: int = 16,
        max_wait_ms: int = 8,
        max_new_tokens: int = 256,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_batch = max_batch
        self._max_wait_ms = max_wait_ms
        self._max_new_tokens = max_new_tokens

        self._model = None
        self._tok = None
        self._loaded = asyncio.Event()

        self._queue: "asyncio.Queue[_PendingReq]" = asyncio.Queue()
        self._batch_task: Optional[asyncio.Task] = None

    # ── Загрузка / прогрев ───────────────────────────────────────────────────
    async def warmup(self) -> None:
        if self._loaded.is_set():
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_blocking)
        self._loaded.set()
        # Стартуем фоновый батчер.
        if self._batch_task is None:
            self._batch_task = asyncio.create_task(self._batch_loop())
        # Прогрев генерации.
        await self.translate("warm up", "en", "ru", beam_size=1, context=[])
        logger.info("M2MEngine прогрет (%s на %s)", self._model_name, self._device)

    def _load_blocking(self) -> None:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        self._tok = M2M100Tokenizer.from_pretrained(self._model_name)
        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._model = (
            M2M100ForConditionalGeneration.from_pretrained(self._model_name, dtype=dtype)
            .to(self._device)
            .eval()
        )

    # ── Публичный перевод (через батчер) ─────────────────────────────────────
    async def translate(
        self, text: str, src: str, tgt: str, *, beam_size: int, context: list[str]
    ) -> str:
        if not text.strip():
            return ""
        if not self._loaded.is_set():
            raise RuntimeError("NMT-движок не прогрет.")
        # context пока не инжектим в модель (см. docstring); резерв под context-aware NMT.
        req = _PendingReq(text=text, src=src, tgt=tgt, beam=beam_size)
        await self._queue.put(req)
        return await req.future

    # ── Фоновый динамический батчер ──────────────────────────────────────────
    async def _batch_loop(self) -> None:
        while True:
            try:
                first = await self._queue.get()
                batch = [first]
                # Добираем окно.
                deadline = time.monotonic() + self._max_wait_ms / 1000
                while len(batch) < self._max_batch:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                    except asyncio.TimeoutError:
                        break
                await self._process_batch(batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Сбой батч-цикла NMT")

    async def _process_batch(self, batch: list[_PendingReq]) -> None:
        # Группируем по (src, tgt, beam) — у каждой группы свой forced_bos и beam.
        groups: dict[tuple[str, str, int], list[_PendingReq]] = {}
        for r in batch:
            groups.setdefault((r.src, r.tgt, r.beam), []).append(r)

        loop = asyncio.get_running_loop()
        for (src, tgt, beam), reqs in groups.items():
            texts = [r.text for r in reqs]
            try:
                outputs = await loop.run_in_executor(
                    None, self._generate_blocking, texts, src, tgt, beam
                )
                for r, out in zip(reqs, outputs):
                    if not r.future.done():
                        r.future.set_result(out)
            except Exception as e:  # noqa: BLE001
                for r in reqs:
                    if not r.future.done():
                        r.future.set_exception(e)

    def _generate_blocking(self, texts: list[str], src: str, tgt: str, beam: int) -> list[str]:
        import torch

        self._tok.src_lang = src
        enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True).to(self._device)
        with torch.no_grad():
            gen = self._model.generate(
                **enc,
                forced_bos_token_id=self._tok.get_lang_id(tgt),
                num_beams=beam,
                no_repeat_ngram_size=3,
                max_new_tokens=self._max_new_tokens,
            )
        return [s.strip() for s in self._tok.batch_decode(gen, skip_special_tokens=True)]

    async def aclose(self) -> None:
        if self._batch_task is not None:
            self._batch_task.cancel()