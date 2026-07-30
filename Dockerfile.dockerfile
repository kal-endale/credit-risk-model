FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV MODEL_PATH=/app/models/best_model.joblib

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements-api.txt .

RUN python -m pip install --no-cache-dir \
    --upgrade pip \
    && python -m pip install --no-cache-dir \
    -r requirements-api.txt

COPY src ./src
COPY models/best_model.joblib ./models/best_model.joblib

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
    --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]