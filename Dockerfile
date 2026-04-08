# ---- Build stage --------------------------------------------------------
FROM python:3.11-slim AS base

LABEL maintainer="AI Systems Engineer"
LABEL description="EmailTriageEnv — OpenEnv-compatible RL environment"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps --------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application code ---------------------------------------------------
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 agentuser && chown -R agentuser:agentuser /app
USER agentuser

# ---- Environment variables ----------------------------------------------
# HF_TOKEN must be injected at runtime via -e flag or a .env file.
# Option 1 — environment variable:
#   docker run -e HF_TOKEN=hf_... email_triage_env
# Option 2 — .env file (copy .env.example → .env and fill in your token):
#   docker run --env-file .env email_triage_env
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ---- Default entrypoint -------------------------------------------------
# Run the FastAPI server by default on the Hugging Face Spaces port
EXPOSE 7860

ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["server:app", "--host", "0.0.0.0", "--port", "7860"]

# ---- Health check -------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# ---- Usage examples (in comments) --------------------------------------
# Build:
#   docker build -t email_triage_env .
#
# Run API server (default):
#   docker run -e HF_TOKEN=$HF_TOKEN -p 8000:8000 email_triage_env
#
# Run inference script:
#   docker run -e HF_TOKEN=$HF_TOKEN --entrypoint python email_triage_env \
#     inference.py
#
# Using .env file:
#   docker run --env-file .env -p 8000:8000 email_triage_env
#
# Run tests:
#   docker run --entrypoint pytest email_triage_env tests/ -v
