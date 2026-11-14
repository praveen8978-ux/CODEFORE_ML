FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cfrec ./cfrec
COPY static ./static
# Train locally first (python -m cfrec.train) and commit artifacts/model.joblib.
COPY artifacts/model.joblib ./artifacts/model.joblib

ENV CFREC_ARTIFACT=/app/artifacts/model.joblib
EXPOSE 8000
CMD ["sh", "-c", "uvicorn cfrec.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
