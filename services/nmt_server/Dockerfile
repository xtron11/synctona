# services/nmt_server/Dockerfile
# Контекст сборки = ./services/nmt_server
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

WORKDIR /app

# curl — для healthcheck из docker-compose.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./nmt_server/requirements.txt

# Объявляем дефолтный индекс для CPU
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} torch \
 && pip install -r ./nmt_server/requirements.txt

COPY . ./nmt_server
ENV PYTHONPATH=/app

EXPOSE 8002
CMD ["uvicorn", "nmt_server.app:app", "--host", "0.0.0.0", "--port", "8002"]