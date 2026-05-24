# --- Stage 1: Build React Frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /frontend

# Copy npm configuration files and install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Copy React codebase and build the static bundle
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Setup FastAPI Backend ---
FROM python:3.10-slim

# Install system-level binaries needed for audio processing and layout tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy FastAPI backend code
COPY backend/ ./backend/

# Copy compiled frontend from stage 1 to backend serving folder
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# Setup writable directory for media/file downloads (required for Hugging Face Spaces UID 1000 execution)
RUN mkdir -p /app/backend/downloads && chmod -R 777 /app

# Open the standard network port that uvicorn communicates through
EXPOSE 7860

# Launch uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]