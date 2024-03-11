echo "Starting API container..."
docker compose -f docker-compose.prod.gpu.yaml down -t 1
docker compose -f docker-compose.prod.gpu.yaml up