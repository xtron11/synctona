# Synctona

MVP системы **синхронного перевода речи в реальном времени с клонированием голоса**. Человек говорит на одном языке, собеседник почти сразу слышит перевод на другом — голосом говорящего.

Пайплайн: **ASR → NMT → TTS** поверх [LiveKit Agents](https://docs.livekit.io/agents/) (WebRTC).

```
 микрофон ──► LiveKit ──► agent ──► ASR (faster-whisper + Silero VAD)
                            │
                            ├──► NMT (M2M-100 418M)
                            │
 динамик  ◄── LiveKit ◄─────┴──► TTS (Chatterbox, клон голоса)
```

## Ключевые идеи

- **Конечный автомат турна**: `LISTENING → TRANSLATING → SPEAKING → BARGE_IN`.
- **Barge-in**: если собеседник перебивает, текущая озвучка и хвост перевода атомарно отменяются (включая GPU-джобу TTS).
- **Пайплайнинг предложений**: следующее предложение переводится и синтезируется, пока текущее ещё звучит.
- **Лестница деградации** под нагрузкой: полный клон-голос → быстрый TTS → generic-голос → только субтитры.
- **Наблюдаемость**: метрики Prometheus, трейсинг OpenTelemetry, готовый дашборд Grafana.

## Структура

```
src/synctona/
  agent_worker.py      # оркестратор сессии на LiveKit Agents
  config.py            # вся конфигурация (pydantic-settings, префикс SYNCTONA_)
  clients/             # клиенты ASR / NMT / TTS и реестр голосов (Redis)
  pipeline/            # сегментация на предложения, чанкинг, темп речи
  observability/       # метрики и трейсинг
services/
  asr_server/          # стриминговое распознавание (WebSocket /v1/stream)
  nmt_server/          # перевод (POST /v1/translate)
  tts_server/          # синтез с клонированием (WebSocket /v1/synthesize)
  enrollment/          # запись образца голоса с согласием (POST /v1/enroll)
scripts/
  smoke_test.py        # прогон цепочки ASR → NMT → TTS без LiveKit
  loadtest.py          # нагрузочный тест
  enroll.py            # регистрация голоса
deploy/observability/  # Prometheus + Grafana
docker/                # Dockerfile агента
tests/                 # интеграционные тесты
```

## Запуск

Нужен Docker (для GPU — драйвер NVIDIA и Docker Desktop с WSL2).

```bash
# CPU
docker compose up --build

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# + регистрация голосов и наблюдаемость
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voices.yml -f docker-compose.observability.yml up -d --build
```

Первый старт долгий: сервисы скачивают модели (кэшируются в docker-томах).

Порты: LiveKit `7880`, TTS `8001`, NMT `8002`, ASR `8003`, Redis `6379`. С observability: Prometheus `9090`, Grafana `3000`, Jaeger `16686`.

### Проверка без LiveKit

```bash
pip install aiohttp numpy soundfile
python scripts/smoke_test.py --text "Привет, как дела?" --src ru --tgt en
python scripts/smoke_test.py --wav sample.wav --src ru --tgt en
```

## Конфигурация

Все параметры задаются переменными окружения с префиксом `SYNCTONA_` (вложенность через `__`), например `SYNCTONA_LIVEKIT__URL`. Пример — в [`.env.example`](.env.example), полный список — в `src/synctona/config.py`.

## Стек

Python 3.10+, LiveKit Agents 1.x, FastAPI, faster-whisper, Silero VAD, M2M-100 (transformers), Chatterbox TTS, Redis, Prometheus, OpenTelemetry, Grafana, Docker Compose.

## Лицензия

MIT
