"""
Synctona — калибровка ёмкости TTS-пула (Step 2, закрывает §5.4 ТЗ).

Гоняет TTS-пул на растущей конкурентности и ищет точку, где TTFB p95 пробивает
бюджет. Это даёт РЕАЛЬНЫЙ max_streams_per_gpu вместо гипотезы из config.py.

Запуск:
    python -m scripts.loadtest --url ws://localhost:8001 --budget-ms 700 \
        --levels 1,2,4,8,16,24,32 --requests-per-level 40

Результат: таблица (concurrency → ttfb p50/p95, rtf p50) и рекомендация:
максимальная конкурентность, при которой ttfb_p95 ≤ бюджет.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from dataclasses import dataclass
import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import aiohttp

SAMPLE_TEXT = (
    "The project was supposed to be a breakthrough, an under ice base in Antarctica "
    "where anomalies had confused scientists for decades."
)


@dataclass
class Result:
    ttfb_ms: float
    rtf: float
    ok: bool
    rejected: bool


async def _one_request(session: aiohttp.ClientSession, ws_url: str, sr: int = 24000) -> Result:
    job_id = f"loadtest:{uuid.uuid4().hex[:12]}"
    t_start = time.monotonic()
    first_ts = None
    audio_bytes = 0
    rejected = False
    try:
        async with session.ws_connect(f"{ws_url}/v1/synthesize", heartbeat=15) as ws:
            await ws.send_json(
                {"job_id": job_id, "text": SAMPLE_TEXT, "voice_id": "default",
                 "preset": "default", "sample_rate": sr}
            )
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if first_ts is None:
                        first_ts = time.monotonic()
                    audio_bytes += len(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data).get("event")
                    if event == "rejected":
                        rejected = True
                        break
                    if event in ("eof", "error"):
                        break
    except Exception:
        return Result(ttfb_ms=0.0, rtf=0.0, ok=False, rejected=rejected)

    if first_ts is None:
        return Result(ttfb_ms=0.0, rtf=0.0, ok=False, rejected=rejected)

    ttfb_ms = (first_ts - t_start) * 1000
    audio_s = (audio_bytes / 2) / sr
    rtf = (time.monotonic() - t_start) / audio_s if audio_s > 0 else 0.0
    return Result(ttfb_ms=ttfb_ms, rtf=rtf, ok=True, rejected=rejected)


async def _run_level(ws_url: str, concurrency: int, total: int) -> list[Result]:
    results: list[Result] = []
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(concurrency)

        async def _task() -> None:
            async with sem:
                results.append(await _one_request(session, ws_url))

        await asyncio.gather(*[_task() for _ in range(total)])
    return results


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = int(round((len(values) - 1) * q))
    return values[k]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8001")
    ap.add_argument("--budget-ms", type=float, default=700.0)
    ap.add_argument("--levels", default="1,2,4,8,16,24,32")
    ap.add_argument("--requests-per-level", type=int, default=40)
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    print(f"{'conc':>5} {'ok':>4} {'rej':>4} {'ttfb_p50':>9} {'ttfb_p95':>9} {'rtf_p50':>8}")
    print("-" * 46)

    best_within_budget = 0
    for conc in levels:
        res = await _run_level(args.url, conc, args.requests_per_level)
        ok = [r for r in res if r.ok]
        rej = sum(1 for r in res if r.rejected)
        ttfbs = [r.ttfb_ms for r in ok]
        rtfs = [r.rtf for r in ok]
        p50 = _p(ttfbs, 0.50)
        p95 = _p(ttfbs, 0.95)
        rtf50 = statistics.median(rtfs) if rtfs else 0.0
        print(f"{conc:>5} {len(ok):>4} {rej:>4} {p50:>9.0f} {p95:>9.0f} {rtf50:>8.3f}")
        if ok and p95 <= args.budget_ms and rej == 0:
            best_within_budget = conc

    print("-" * 46)
    print(f"Рекомендация max_streams_per_gpu ≈ {best_within_budget} "
          f"(макс. конкурентность при ttfb_p95 ≤ {args.budget_ms:.0f}ms без отказов)")
    print("Подставь это значение в SYNCTONA_TTS_POOL__MAX_STREAMS_PER_GPU "
          "и пересчитай GPUs_tts по формуле §5.")


if __name__ == "__main__":
    asyncio.run(main())