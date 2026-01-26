# Use Python 3.10 slim image as base (compatible with librosa and dependencies)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-test.txt ./

# Install Python dependencies without version pinning issues
RUN pip install --upgrade pip && \
    pip install librosa numpy scikit-learn joblib tqdm && \
    pip install pytest pytest-cov pytest-mock soundfile

# Copy project files
COPY src/ ./src/
COPY tests/ ./tests/
COPY pytest.ini ./

# Create necessary directories
RUN mkdir -p data/female data/male artifacts feedback_data logs

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "src/main.py"]