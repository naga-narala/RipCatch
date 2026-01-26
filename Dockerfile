# ============================================================================
# RipCatch v2.0 - Production Dockerfile
# Multi-stage build for optimized image size
# Supports both CPU and GPU inference
# ============================================================================

# ============================================================================
# Stage 1: Base Image with Dependencies
# ============================================================================
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base AS dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

# Copy application files
COPY app.py .
COPY RipCatch-v2.0/Model/weights/best.pt RipCatch-v2.0/Model/weights/

# Create directory for uploads (if needed)
RUN mkdir -p /app/uploads /app/outputs

# Expose Gradio default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Set user for security (non-root)
RUN useradd -m -u 1000 ripcatch && \
    chown -R ripcatch:ripcatch /app
USER ripcatch

# Run the application
CMD ["python", "app.py"]
