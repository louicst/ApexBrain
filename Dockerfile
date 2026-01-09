# ---------------------------------------
# STAGE 1: BUILDER (Compiling Dependencies)
# ---------------------------------------
FROM python:3.10-slim as builder

WORKDIR /app

# Install system compilers (needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create wheels (compiled packages)
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# ---------------------------------------
# STAGE 2: RUNNER (Final Runtime Image)
# ---------------------------------------
FROM python:3.10-slim

WORKDIR /app

# Metadata
LABEL maintainer="ApexBrain Engineering"
LABEL version="2.0.0-Enterprise"

# Install minimal runtime libs
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies from wheels
RUN pip install --no-cache /wheels/*

# Copy Source Code
COPY . .

# Create cache directory for FastF1
RUN mkdir -p /app/cache

# Create non-root user for security (Pro Practice)
RUN useradd -m apexuser
RUN chown -R apexuser:apexuser /app
USER apexuser

# Healthcheck endpoint for orchestrators (Kubernetes/AWS)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Expose Port
EXPOSE 8501

# Launch Command
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--theme.base=light"]