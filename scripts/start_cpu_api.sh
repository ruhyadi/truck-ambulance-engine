echo "Starting API container..."
docker compose -f docker-compose.prod.cpu.yaml down -t 1
docker compose -f docker-compose.prod.cpu.yaml up