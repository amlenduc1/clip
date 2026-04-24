@echo off
REM Batch script to build and start the full SEM defect classification stack on Docker Desktop

REM Build and start all services
cd /d %~dp0

echo Building and starting all containers...
docker compose up --build -d

echo "All services are starting."
echo "Wait a few moments, then open http://localhost:8501 for the Streamlit UI."
echo "To index images, place them in backend/sem_images and run:"
echo "docker compose exec backend python index_images.py"
