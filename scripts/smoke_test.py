"""
Synctona — дымовой тест каскада (без LiveKit).

Гоняет реальные сервисы цепочкой ASR → NMT → TTS:
  • --wav FILE   : распознать речь из файла, перевести, озвучить перевод в --out
  • --text "..." : пропустить ASR, только перевести и озвучить (быстрая проверка NMT+TTS)

Запуск (на хосте, где подняты контейнеры):
    pip install aiohttp numpy soundfile
    python scripts/smoke_test.py --wav my_voice.wav --src ru --tgt en
    python scripts/smoke_test.py --text "Привет, как дела?" --src ru --tgt en

TTS здесь работает generic-голосом (voice=default): клонирование требует, чтобы
образец голоса лежал внутри TTS-контейнера — это отдельный флоу (см. «что осталось»).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave

import aiohttp
import numpy as np

ASR = "http://localhost:8003"
NMT = "http://localhost:8002"
TTS = "http://localhost:8001"


# ─────────────────────────── Аудио-хелперы ──────────────────────────────────
def load_wav_16k_mono(path: str) -> bytes:
    """Читает wav, приводит к 16 кГц моно int16 (то, что ждёт ASR)."""
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception:
        with wave.open(path, "rb") as w:
            sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
            raw = w.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)

    if sr != 16000:
        tgt_len = int(len(data) / sr * 16000)
        data = np.interp(
            np.linspace(0, len(data), tgt_len, endpoint=False),
            np.arange(len(data)),
            data,
        ).astype(np.float32)

    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def save_wav(path: str, pcm16: bytes, sample_rate: int = 24000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)


# ─────────────────────────── Стадии ─────────────────────────────────────────
async def run_asr(session: aiohttp.ClientSession, pcm: bytes, language: str) -> str:
    ws_url = ASR.replace("http", "ws") + "/v1/stream"
    finals: list[str] = []
    async with session.ws_connect(ws_url) as ws:
        await ws.send_json({"session_id": "smoke", "language": language, "sample_rate": 16000})
        # Шлём аудио порциями ~200 мс, имитируя поток.
        step = 3200 * 2  # 3200 сэмплов * 2 байта
        for i in range(0, len(pcm), step):
            await ws.send_bytes(pcm[i:i + step])
            await asyncio.sleep(0.02)
        await ws.send_str(json.dumps({"event": "finalize"}))

        # Дочитываем события до тишины (ASR держит сокет открытым).
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=3.0)
            except asyncio.TimeoutError:
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                ev = json.loads(msg.data)
                if ev.get("type") == "partial":
                    print(f"  · partial: {ev['text']}")
                elif ev.get("type") == "final":
                    finals.append(ev["text"])
                    print(f"  ✓ final:   {ev['text']}")
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
    return " ".join(finals).strip()


async def run_nmt(session: aiohttp.ClientSession, text: str, src: str, tgt: str) -> str:
    async with session.post(f"{NMT}/v1/translate",
                            json={"text": text, "src": src, "tgt": tgt, "beam": 4}) as r:
        r.raise_for_status()
        return (await r.json())["translation"]


async def run_tts(session, text: str, voice: str = "default") -> bytes:
    ws_url = TTS.replace("http", "ws") + "/v1/synthesize"
    pcm = bytearray()
    async with session.ws_connect(ws_url) as ws:
        await ws.send_json({"job_id": "smoke", "text": text, "voice_id": voice,
                            "preset": "default", "sample_rate": 24000})
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                pcm += msg.data
            elif msg.type == aiohttp.WSMsgType.TEXT:
                if json.loads(msg.data).get("event") in ("eof", "error", "rejected"):
                    break
    return bytes(pcm)


# ─────────────────────────── main ───────────────────────────────────────────
async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="Исходный wav с речью (для полного каскада)")
    ap.add_argument("--text", help="Текст напрямую (пропустить ASR)")
    ap.add_argument("--src", default="ru")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--out", default="tts_output.wav")
    ap.add_argument("--voice", default="default")
    args = ap.parse_args()

    if not args.wav and not args.text:
        ap.error("укажи --wav или --text")

    async with aiohttp.ClientSession() as session:
        if args.wav:
            print(f"[ASR] распознаю {args.wav} ...")
            pcm = load_wav_16k_mono(args.wav)
            source = await run_asr(session, pcm, args.src)
            print(f"[ASR] итог: {source!r}")
        else:
            source = args.text
            print(f"[SRC] {source!r}")

        if not source:
            print("Пустой текст после ASR — нечего переводить.")
            return

        print("[NMT] перевожу ...")
        translation = await run_nmt(session, source, args.src, args.tgt)
        print(f"[NMT] перевод: {translation!r}")

        print("[TTS] озвучиваю ...")
        audio = await run_tts(session, translation, voice=args.voice)
        save_wav(args.out, audio, 24000)
        print(f"[TTS] сохранил {args.out} ({len(audio) // 2 / 24000:.1f} c) — послушай.")


if __name__ == "__main__":
    asyncio.run(main())