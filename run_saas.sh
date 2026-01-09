#!/bin/bash

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   🏎️  APEXBRAIN ENTERPRISE LAUNCHER    ${NC}"
echo -e "${BLUE}=========================================${NC}"

# 1. Check Docker
if ! docker info > /dev/null 2>&1; then
  echo "❌ Error: Docker is not running."
  echo "👉 Please open Docker Desktop and try again."
  exit 1
fi

# 2. Build
echo -e "${GREEN}🔨 Building Container Image (apexbrain:latest)...${NC}"
docker build -t apexbrain:latest .

# 3. Run
echo -e "${GREEN}🚀 Deploying to Localhost:8501...${NC}"
echo "Press Ctrl+C to stop the server."
docker run -p 8501:8501 apexbrain:latest