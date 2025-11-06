FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install poetry
RUN pip install poetry

# Configure poetry to not create virtual env
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-root

# Copy application code
COPY recommender ./recommender
COPY config.yaml ./

# Create data directories
RUN mkdir -p data cache models config

# Expose port
EXPOSE 8080

# Default command
CMD ["uvicorn", "recommender.api.server:app", "--host", "0.0.0.0", "--port", "8080"]
