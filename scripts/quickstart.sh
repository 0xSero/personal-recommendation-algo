#!/bin/bash
set -e

echo "🚀 Personal Recommender Quick Start"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose first."
    exit 1
fi

echo "✓ Docker found"

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p data cache models config

# Check if config files exist
if [ ! -f "config.yaml" ]; then
    echo "❌ config.yaml not found. Please ensure it exists."
    exit 1
fi

echo "✓ Directories created"

# Start services
echo ""
echo "🐳 Starting Docker services..."
docker-compose up -d qdrant

echo "⏳ Waiting for Qdrant to be ready..."
sleep 5

echo ""
echo "🤖 Starting vLLM (this may take a few minutes to download model)..."
echo "   Note: This will download ~90GB for Mixtral-8x7B"
docker-compose up -d vllm

echo "⏳ Waiting for vLLM to load model..."
echo "   (This can take 5-10 minutes on first run)"

# Wait for vLLM
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    echo -n "."
    sleep 10
done

echo ""
echo "✓ vLLM ready"

# Start API
echo ""
echo "🌐 Starting API server..."
docker-compose up -d api

echo "⏳ Waiting for API to be ready..."
sleep 5

until curl -s http://localhost:8080/ > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done

echo ""
echo ""
echo "✅ All services started successfully!"
echo ""
echo "📍 Service URLs:"
echo "   - API:    http://localhost:8080"
echo "   - Qdrant: http://localhost:6333"
echo "   - vLLM:   http://localhost:8000"
echo ""
echo "📖 Next steps:"
echo "   1. Add RSS feeds: edit config/rss_feeds.txt"
echo "   2. Run ingestion: python scripts/ingest.py"
echo "   3. Get recommendations: curl http://localhost:8080/recommend"
echo "   4. Try example client: python examples/client_example.py"
echo ""
echo "📊 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Stop services:"
echo "   docker-compose down"
