# ─────────────────────────────────────────────────────────────────────────────
# Tech Office Cockpit — Dockerfile multi-stage
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1 : Builder — Installation des dépendances
FROM python:3.11-slim AS builder

WORKDIR /app

# Dépendances système pour Unstructured.io / Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    poppler-utils \
    libmagic1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2 : Runner — Image finale légère
FROM python:3.11-slim

WORKDIR /app

# Copier uniquement les dépendances installées
COPY --from=builder /install /usr/local

# Dépendances runtime (Tesseract, poppler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root pour la sécurité
RUN groupadd -r techoffice && useradd -r -g techoffice techoffice
USER techoffice

# Copier le code source
COPY --chown=techoffice:techoffice src/ ./src/

# Variables d'environnement par défaut (surchargées par K8s Secrets)
ENV APP_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health/').raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
