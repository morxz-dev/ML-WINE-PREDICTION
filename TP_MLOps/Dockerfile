FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps -r requirements.txt --wheel-dir /wheels

FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --find-links /wheels -r requirements.txt

COPY src ./src

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

CMD ["uvicorn", "mlops_tp.api:app", "--host", "0.0.0.0", "--port", "8000"]
