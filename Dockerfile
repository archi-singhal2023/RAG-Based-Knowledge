# ── Stage 1: Base image ───────────────────────────────────────────────────────
# Python 3.11 slim keeps the image small while having everything we need
FROM python:3.11-slim

# ── Stage 2: System dependencies ──────────────────────────────────────────────
# build-essential: needed to compile some Python packages (chromadb, tokenizers)
# libgomp1: required by sentence-transformers for parallel processing
# curl: useful for health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 3: Working directory ────────────────────────────────────────────────
WORKDIR /app

# ── Stage 4: Install Python dependencies ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 5: Download embedding model at build time ───────────────────────────
# all-MiniLM-L6-v2 is ~90MB. Downloading during build means:
# - First request is not slow (model already on disk)
# - Works offline inside the container
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Stage 6: Copy application code ────────────────────────────────────────────
COPY . .

# ── Stage 7: Runtime setup ────────────────────────────────────────────────────
RUN mkdir -p chroma_db_sessions

# Tell Python not to write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Tell Python not to buffer stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1

# ── Stage 8: Port ─────────────────────────────────────────────────────────────
EXPOSE 8080

# ── Stage 9: Start command ────────────────────────────────────────────────────
# --workers 1: important — global state means multiple workers cause bugs
# --timeout 120: PDF processing + LLM calls can take >30s
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]