.PHONY: help install dev-install run-docker stop-docker logs ingest test clean

help:
	@echo "Personal Recommender - Make Commands"
	@echo ""
	@echo "  make install        Install dependencies with poetry"
	@echo "  make dev-install    Install with dev dependencies"
	@echo "  make run-docker     Start all services with docker-compose"
	@echo "  make stop-docker    Stop all services"
	@echo "  make logs           Follow docker logs"
	@echo "  make ingest         Run ingestion pipeline"
	@echo "  make test           Run tests"
	@echo "  make clean          Clean generated files"

install:
	poetry install --no-dev

dev-install:
	poetry install

run-docker:
	docker-compose up -d
	@echo ""
	@echo "✅ Services started!"
	@echo "   Qdrant:  http://localhost:6333"
	@echo "   vLLM:    http://localhost:8000"
	@echo "   API:     http://localhost:8080"

stop-docker:
	docker-compose down

logs:
	docker-compose logs -f

ingest:
	python scripts/ingest.py

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/
