# Build: docker build -t f1-top10-predictor .
# Run:   docker run -p 8000:8000 f1-top10-predictor
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ ./src/
COPY models/ ./models/
COPY mlflow.db .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
