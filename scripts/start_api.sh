echo "Starting API container..."
docker compose -f docker-compose.prod.api.yaml down -t 1
docker compose -f docker-compose.prod.api.yaml up