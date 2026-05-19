FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install system dependencies for OpenCV, FFmpeg, and uv git dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --extra test --extra dev

# Copy the rest of the backend code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command: run the server
CMD ["uv", "run", "python", "ltx2_server.py"]
