# services/tts_server/Dockerfile
# Контекст сборки = ./services/tts_server.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

WORKDIR /app

# ffmpeg + libsndfile1 нужны для работы с аудио; curl — healthcheck; git — для моделей.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl git ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./tts_server/requirements.txt

# --- ДОБАВЛЕНО ---
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} torch torchaudio \
 && pip install -r ./tts_server/requirements.txt
# ------------------

COPY . ./tts_server
ENV PYTHONPATH=/app

EXPOSE 8001
CMD ["uvicorn", "tts_server.app:app", "--host", "0.0.0.0", "--port", "8001"]