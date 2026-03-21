FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first to avoid huge CUDA packages
RUN pip install --no-cache-dir torch==2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set cache paths BEFORE downloading model
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/models

# Download and cache model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"


COPY . .

RUN mkdir -p chroma_db_sessions

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "--graceful-timeout", "300"]