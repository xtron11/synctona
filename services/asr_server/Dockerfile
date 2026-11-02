# services/asr_server/Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl git libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./asr_server/requirements.txt

# --- ВОТ ТУТ ПРАВКА ---
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} torch torchaudio \
 && pip install -r ./asr_server/requirements.txt
# ----------------------

COPY . ./asr_server
ENV PYTHONPATH=/app

EXPOSE 8003
CMD ["uvicorn", "asr_server.app:app", "--host", "0.0.0.0", "--port", "8003"]