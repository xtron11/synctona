"""
Synctona — клиентский помощник записи голоса (Step: клонирование).

Регистрирует твой голос: шлёт wav-образец + consent в сервис enrollment, получает
voice_id. Дальше этим voice_id можно синтезировать склонированный голос.

    pip install requests
    python scripts/enroll.py --identity ivan --wav my_voice.wav

Важно: --identity должен совпадать с тем, под которым ты заходишь в LiveKit-комнату,
иначе агент не найдёт твой голос по consent. Для проверки через smoke_test это
просто любая строка, которую потом передашь в --voice.
"""

from __future__ import annotations

import argparse

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", required=True, help="ID пользователя (как в LiveKit)")
    ap.add_argument("--wav", required=True, help="Образец голоса, 10–15с чистой речи")
    ap.add_argument("--url", default="http://localhost:8004")
    args = ap.parse_args()

    with open(args.wav, "rb") as f:
        resp = requests.post(
            f"{args.url}/v1/enroll",
            data={"identity": args.identity, "consent": "true"},
            files={"audio": (args.wav, f, "audio/wav")},
            timeout=120,
        )
    if resp.status_code != 200:
        print(f"Ошибка {resp.status_code}: {resp.text}")
        return
    body = resp.json()
    print("Голос зарегистрирован.")
    print(f"  identity : {body['identity']}")
    print(f"  voice_id : {body['voice_id']}")
    print(f"\nПроверить озвучку своим голосом:")
    print(f"  python scripts/smoke_test.py --text \"Hello, this is my cloned voice.\" "
          f"--src en --tgt en --voice {body['voice_id']}")


if __name__ == "__main__":
    main()