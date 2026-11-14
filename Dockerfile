FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cfrec ./cfrec
COPY static ./static
# Train first (python -m cfrec.train) and make sure artifacts/model.joblib exists before building.
COPY artifacts ./artifacts
ENV CFREC_ARTIFACT=/app/artifacts/model.joblib
EXPOSE 8000
CMD ["sh", "-c", "uvicorn cfrec.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
