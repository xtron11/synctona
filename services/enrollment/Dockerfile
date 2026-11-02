# services/enrollment/Dockerfile
# Крошечный сервис: ни моделей, ни torch. Просто оркестрация consent + upload.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN pip install fastapi "uvicorn[standard]" aiohttp redis python-multipart

COPY . ./enrollment
ENV PYTHONPATH=/app

EXPOSE 8004
CMD ["uvicorn", "enrollment.app:app", "--host", "0.0.0.0", "--port", "8004"]