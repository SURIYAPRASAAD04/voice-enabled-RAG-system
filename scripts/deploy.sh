#!/bin/bash
set -e

echo "=== HH Goa 2026 Task #2: Deploying Stack ==="

# Load environment variables
if [ -f .env ]; then
  export $(echo $(grep -v '^#' .env | xargs) | envsubst)
fi

echo "1. Building Docker Images..."
docker compose build

echo "2. Bootstrapping Qdrant & Redis in the background..."
docker compose up -d qdrant redis

echo "3. Waiting for Qdrant Vector DB to initialize..."
until curl -s http://localhost:6333/collections > /dev/null; do
  echo "Waiting for Qdrant to start..."
  sleep 2
done

echo "4. Checking if dataset index exists in Qdrant..."
INDEX_EXISTS=$(curl -s http://localhost:6333/collections/msmarco_chunks | grep -q "msmarco_chunks" && echo "yes" || echo "no")

if [ "$INDEX_EXISTS" = "yes" ]; then
  echo "Vector index 'msmarco_chunks' already exists. Skipping offline ingestion."
else
  echo "Vector index not found. Initiating offline ingestion script..."
  # Run ingestion inside the backend container (starts in workspace context)
  docker compose run --entrypoint "python backend/scripts/ingest.py 30 hi" backend
fi

echo "5. Spin up Frontend and Backend web servers..."
docker compose up -d

echo "6. Running health check..."
sleep 5
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health || echo "failed")

if [ "$HEALTH_STATUS" = "200" ]; then
  echo "=== SUCCESS: Stack is healthy and running! ==="
  echo "Frontend: http://localhost:3000"
  echo "Backend API: http://localhost:8000"
else
  echo "=== WARNING: Health check returned status $HEALTH_STATUS. Check logs with 'docker compose logs' ==="
fi
