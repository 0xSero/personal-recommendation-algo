# Quick Start Guide

Get your personal recommendation system running in 10 minutes.

## Prerequisites

- **Docker & Docker Compose** installed
- **6×3090 GPUs** (or adjust config for your hardware)
- **100GB+ free disk space** (for models)
- **32GB+ RAM**

## 5-Minute Setup

### 1. Clone and Configure

```bash
git clone <repo-url>
cd personal-recommendation-algo

# Edit config.yaml if needed (optional for first run)
```

### 2. Start Services

```bash
# Automated start (recommended)
./scripts/quickstart.sh

# Or manually
docker-compose up -d
```

This will:
- Start Qdrant (vector DB)
- Download and start Mixtral-8×7B via vLLM (~90GB download, one-time)
- Start the API server

**First run takes ~10 minutes** to download the model.

### 3. Add Content

```bash
# Add some RSS feeds
cat >> config/rss_feeds.txt << EOF
https://news.ycombinator.com/rss
https://arxiv.org/rss/cs.AI
EOF

# Run ingestion
python scripts/ingest.py
```

This will:
- Fetch RSS items
- Generate summaries with LLM
- Create embeddings
- Store in vector DB

### 4. Get Recommendations

```bash
# Using CLI (prettiest)
python scripts/cli.py recommend -n 10

# Using curl
curl http://localhost:8080/recommend

# Using Python client
python examples/client_example.py
```

## What Just Happened?

1. **Qdrant** stores vector embeddings of your content
2. **vLLM** serves Mixtral-8×7B for summaries and enrichment
3. **API server** orchestrates recommendations:
   - Generates 500 candidates (vector similarity)
   - Reranks with cross-encoder + priors
   - Adds exploration items (Thompson Sampling)
   - Returns top 20

## Next Steps

### Add More Sources

**YouTube:**
```python
# Implement YouTube ingester (example provided)
# Add to config.yaml:
sources:
  youtube:
    enabled: true
    channels:
      - "https://youtube.com/@channel"
```

**Podcasts:**
```bash
# Add podcast feeds
echo "https://feeds.example.com/podcast" >> config/podcast_feeds.txt

# Ingest (includes transcription with Whisper)
python scripts/ingest.py
```

**Browser History:**
```yaml
# In config.yaml
sources:
  browser_history:
    enabled: true
    chrome_path: "~/.config/google-chrome/Default/History"
```

### Customize Recommendations

**Tune scoring weights** (`config.yaml`):
```yaml
recommendation:
  reranking:
    weights:
      cosine_global: 0.40    # ← Increase for more similar-to-liked
      cosine_session: 0.25   # ← Increase for session context
      cross_encoder: 0.20
      novelty: 0.10          # ← Increase for more diversity
```

**Add your topics** (`config.yaml`):
```yaml
enrichment:
  topics:
    - "Your Topic 1"
    - "Your Topic 2"
    # ... add 20-30 topics you care about
```

### Use Different Models

**Smaller/Faster LLM:**
```yaml
models:
  llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"  # Fits on 2 GPUs
```

**Better Embeddings:**
```yaml
models:
  embedding:
    model_name: "BAAI/bge-m3"  # Multilingual + sparse
```

**Lighter Everything:**
```yaml
models:
  embedding:
    model_name: "thenlper/gte-small"  # 384d, very fast
  reranker:
    model_name: "BAAI/bge-reranker-base"  # Lighter reranker
  llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
```

## Common Tasks

### Get Recommendations with Context

```bash
# 30-minute time budget, prefer articles
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "num_items": 20,
    "context": {
      "time_budget_minutes": 30,
      "preferred_format": "article"
    }
  }'
```

### Log Feedback

```bash
# User completed an item
curl -X POST http://localhost:8080/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "rss_abc123",
    "event_type": "completion",
    "dwell_seconds": 180,
    "completion_fraction": 0.9
  }'
```

### View Statistics

```bash
python scripts/cli.py stats
```

### Check Exploration Arms

```bash
# See which exploration strategies are working
sqlite3 data/recommender.duckdb \
  "SELECT * FROM exploration_arms"
```

## Troubleshooting

### vLLM Out of Memory

**Reduce tensor parallelism:**
```yaml
# docker-compose.yml
command:
  - --tensor-parallel-size
  - "4"  # Instead of 6
```

Or use smaller model:
```yaml
models:
  llm:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
```

### Qdrant Connection Error

```bash
# Check if running
docker ps | grep qdrant

# Restart
docker-compose restart qdrant
```

### Slow Recommendations

- **Reduce candidates**: Set `num_candidates: 200` in config
- **Disable cross-encoder**: Set `cross_encoder: 0.0` in weights
- **Use smaller reranker**: `bge-reranker-base`

### Not Enough Variety

- **Increase exploration**: Set more exploration arm slots
- **Increase novelty weight**: `novelty: 0.10`
- **Lower diversity MMR**: `diversity_lambda: 0.5`

## Performance Tips

### GPU Memory Optimization

```yaml
# docker-compose.yml vLLM service
command:
  - --gpu-memory-utilization
  - "0.85"  # Leave some headroom
  - --max-model-len
  - "4096"  # Reduce if OOM
```

### Batch Processing

```python
# For ingestion, process in batches
embedding_model.encode(texts, batch_size=64)
```

### Caching

```bash
# Cache model weights locally
mkdir -p models
export HF_HOME=./models
```

## Learn More

- **Full README**: [README.md](README.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Docs**: http://localhost:8080/docs (after starting)
- **Config Reference**: [config.yaml](config.yaml)

## Getting Help

1. Check logs: `docker-compose logs -f`
2. Verify services: `docker-compose ps`
3. Test API: `curl http://localhost:8080/`
4. Run tests: `pytest tests/`
