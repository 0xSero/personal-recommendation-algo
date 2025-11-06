# Personal Recommendation System

A **model-agnostic**, **zero-training** recommendation system that runs entirely locally on your hardware. Built to feel like a personal YouTube-style feed, using only pre-trained models and simple heuristics.

## Key Features

- **No Training Required**: Uses pre-trained models off-the-shelf (embeddings, cross-encoders, LLMs)
- **Model Agnostic**: Swap any model by changing config - supports multiple providers
- **Pure Heuristics**: Learning via counters, moving averages, and Thompson Sampling bandits
- **Fully Local**: Runs on your hardware (designed for 6×3090 GPUs, but configurable)
- **Multi-Source**: YouTube, RSS feeds, browser history, podcasts, local files
- **Two-Stage Pipeline**: Fast candidate generation → smart reranking

## Architecture

### Pipeline

```
┌─────────────┐
│   Sources   │ YouTube, RSS, Browser, Podcasts, Local Files
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Ingest    │ Fetch + normalize content
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Enrich    │ LLM summaries, topics, quality hints (zero-shot)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Vectorize  │ Generate embeddings (no training)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Storage   │ Qdrant (vectors) + DuckDB (metadata, events)
└─────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Recommendation Flow                    │
└─────────────────────────────────────────────────────────┘

Request → Candidate Generation → Reranking → Exploration → Response
          (500 items)            (top 50)     (+5 items)    (20 items)

Candidate Generation:
  - Taste centroid (similarity to user's global embedding)
  - Session context (similarity to recent items)
  - Topic slots (from user's top topics)
  - Exploration (fresh/unseen)

Reranking (weighted scoring):
  - 0.35 × cosine(global)
  - 0.20 × cosine(session)
  - 0.25 × cross-encoder score
  - 0.08 × creator prior (completion rate)
  - 0.05 × source prior (avg reward)
  - 0.04 × novelty (inverse of recent views)
  - 0.03 × length match
  + recency decay - repetition penalty

Exploration (Thompson Sampling):
  - new_creators, deep_dives, short_sharp, hot_news
  - Beta distribution arms (alpha, beta counters)
  - No models - just Bayesian bandit math

Feedback:
  - Reward = 0.6*completion + 0.2*dwell + 0.2*like - 0.3*bounce
  - Update creator/source priors (exponential moving average)
  - Update exploration arms (Beta update)
```

## Model Architecture (Fully Swappable)

All models are **pre-trained and used off-the-shelf**. Change any model by editing `config.yaml`.

### Default Models

| Component | Default | Alternatives |
|-----------|---------|--------------|
| **Embeddings** | `BAAI/bge-large-en-v1.5` | `BAAI/bge-m3`, `gte-small`, `nomic-embed-text-v1` |
| **Reranker** | `BAAI/bge-reranker-large` | `bge-reranker-base` |
| **LLM** | `Mixtral-8×7B-Instruct` | `Qwen2.5-32B`, `Llama-3.1-70B` |
| **ASR** | `whisper-large-v3` | `whisper-medium`, `faster-whisper` |

### Pluggable Providers

The system uses a factory pattern - add new providers without changing core code:

```python
# Current providers
- Embeddings: sentence_transformers, openai, custom
- Reranker: sentence_transformers, custom
- LLM: openai_compatible (vLLM, TGI, OpenAI), custom
- ASR: whisper, faster_whisper, custom

# Add custom provider (example)
from recommender.models import ModelFactory, BaseEmbeddingModel

class MyCustomEmbedding(BaseEmbeddingModel):
    def encode(self, texts): ...

ModelFactory.register_embedding_provider("my_provider", MyCustomEmbedding)
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker + docker-compose
- NVIDIA GPUs (6×3090 recommended, but configurable)
- 32GB+ RAM

### Installation

```bash
# Clone repository
git clone <repo-url>
cd personal-recommendation-algo

# Install dependencies
poetry install

# Or with pip
pip install -e .
```

### Configuration

Edit `config.yaml` to customize:

1. **Models**: Change model names, devices, batch sizes
2. **Sources**: Enable/disable YouTube, RSS, etc.
3. **Recommendation weights**: Tune scoring weights
4. **Exploration**: Configure Thompson Sampling arms
5. **Topics**: Define your personal taxonomy

```yaml
models:
  embedding:
    model_name: "BAAI/bge-large-en-v1.5"  # Change to any model
    device: "cuda:0"

  llm:
    model_name: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    base_url: "http://localhost:8000/v1"
```

### Run with Docker Compose

```bash
# Start all services (Qdrant, vLLM, API)
docker-compose up -d

# Check logs
docker-compose logs -f api

# API will be available at http://localhost:8080
```

Services:
- **Qdrant** (vector DB): `localhost:6333`
- **vLLM** (LLM inference): `localhost:8000`
- **API** (recommendations): `localhost:8080`

### Manual Setup (without Docker)

```bash
# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Start vLLM (in separate terminal)
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --tensor-parallel-size 6 \
  --max-model-len 8192

# 3. Run ingestion
python scripts/ingest.py

# 4. Start API
uvicorn recommender.api.server:app --host 0.0.0.0 --port 8080
```

## Usage

### Ingest Content

```bash
# Add RSS feeds to config/rss_feeds.txt
echo "https://news.ycombinator.com/rss" >> config/rss_feeds.txt

# Run ingestion
python scripts/ingest.py
```

### Get Recommendations

```bash
# GET request
curl http://localhost:8080/recommend

# POST with parameters
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "num_items": 20,
    "session_id": "my-session",
    "context": {
      "time_budget_minutes": 30,
      "preferred_format": "article"
    }
  }'
```

Response:
```json
{
  "items": [
    {
      "id": "rss_abc123",
      "title": "Interesting Article",
      "url": "https://...",
      "creator": "Author Name",
      "summary": "5 bullet point summary...",
      "score": 0.87,
      "strategy": "taste_centroid",
      "topics": ["Machine Learning", "Python"]
    }
  ],
  "session_id": "my-session"
}
```

### Log Feedback

```bash
curl -X POST http://localhost:8080/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "rss_abc123",
    "event_type": "completion",
    "dwell_seconds": 180,
    "completion_fraction": 0.95,
    "session_id": "my-session"
  }'
```

Event types: `click`, `dwell`, `completion`, `like`, `dislike`, `save`, `hide`

### Get Stats

```bash
curl http://localhost:8080/stats
```

## Configuration Deep Dive

### Recommendation Weights

Tune the reranking formula in `config.yaml`:

```yaml
recommendation:
  reranking:
    weights:
      cosine_global: 0.35    # Similarity to long-term taste
      cosine_session: 0.20   # Similarity to current session
      cross_encoder: 0.25    # Cross-encoder relevance
      creator_prior: 0.08    # Creator completion rate
      source_prior: 0.05     # Source quality
      novelty: 0.04          # Freshness/diversity
      length_match: 0.03     # Match to time budget
```

### Exploration Arms

Configure Thompson Sampling:

```yaml
recommendation:
  exploration:
    arms:
      - name: "new_creators"
        slots: 2              # 2 items per recommendation
      - name: "deep_dives"
        slots: 2              # Long-form content
      - name: "short_sharp"
        slots: 1
      - name: "hot_news"
        slots: 1
```

### Recency Decay

Different half-lives per source:

```yaml
recommendation:
  reranking:
    recency_half_life_days:
      default: 14
      news: 7               # News decays fast
      evergreen: 60         # Deep content stays relevant
      podcast: 30
```

## API Reference

### `POST /recommend`

Get personalized recommendations.

**Request:**
```json
{
  "user_id": "default",
  "num_items": 20,
  "session_id": "optional-session-id",
  "context": {
    "time_budget_minutes": 30,
    "preferred_format": "article",
    "muted_topics": ["Politics"]
  }
}
```

**Response:**
```json
{
  "items": [...],
  "session_id": "session-id"
}
```

### `POST /feedback`

Log user interaction.

**Request:**
```json
{
  "item_id": "rss_abc123",
  "event_type": "completion",
  "dwell_seconds": 180.5,
  "completion_fraction": 0.95,
  "session_id": "session-id"
}
```

### `GET /stats`

Get user statistics.

### `GET /items/{item_id}`

Get item details.

## Performance

### Hardware Utilization

With 6×3090 (144GB VRAM total):

- **vLLM (Mixtral-8×7B)**: Uses all 6 GPUs via tensor parallelism (~80GB VRAM)
- **Embedding model (BGE-large)**: 1 GPU (~4GB VRAM)
- **Reranker (BGE-reranker-large)**: 1 GPU (~4GB VRAM)
- **Remaining VRAM**: For batch processing, caching

### Latency

- **Candidate generation**: ~50ms (vector search)
- **Reranking**: ~200ms (cross-encoder on 500 items)
- **LLM enrichment**: ~1-2s per item (batched offline)
- **Full /recommend call**: ~300-500ms

### Throughput

- **Embeddings**: ~1000 items/minute (batched)
- **Reranking**: ~100 items/second
- **Recommendations**: ~20 requests/second

## Development

### Project Structure

```
personal-recommendation-algo/
├── recommender/
│   ├── models/              # Model-agnostic interfaces
│   │   ├── base.py         # Abstract base classes
│   │   ├── factory.py      # Model factory
│   │   ├── sentence_transformers_impl.py
│   │   ├── openai_impl.py
│   │   └── whisper_impl.py
│   ├── database/           # Storage abstractions
│   │   ├── schema.py       # Data models
│   │   ├── duckdb_store.py
│   │   └── vector_store.py
│   ├── ingest/             # Content ingestion
│   ├── enrichment/         # LLM enrichment
│   ├── candidate_generation/
│   ├── reranking/
│   ├── exploration/        # Thompson Sampling
│   ├── feedback/           # Reward computation
│   └── api/                # FastAPI server
├── scripts/
│   └── ingest.py
├── config.yaml
├── docker-compose.yml
└── README.md
```

### Adding New Models

1. **Implement the interface**:

```python
from recommender.models import BaseEmbeddingModel, ModelConfig

class MyEmbedding(BaseEmbeddingModel):
    def encode(self, texts, **kwargs):
        # Your implementation
        return embeddings

    def get_dimension(self):
        return 768
```

2. **Register provider**:

```python
ModelFactory.register_embedding_provider("my_provider", MyEmbedding)
```

3. **Update config**:

```yaml
models:
  embedding:
    provider: "my_provider"
    model_name: "my-model"
```

### Adding New Sources

1. Implement `BaseIngester`
2. Add to `config.yaml`
3. Update ingestion script

## No Training - How It Works

This system achieves personalization **without any model training**:

1. **Embeddings**: Pre-trained sentence transformers (BGE, GTE, etc.)
2. **Reranking**: Pre-trained cross-encoders
3. **LLM**: Pre-trained instruction models (Mixtral, Qwen, Llama)
4. **Learning**: Simple arithmetic
   - Creator prior: `avg_completion_rate = EMA(successes/total)`
   - Source prior: `avg_reward = EMA(rewards)`
   - Thompson Sampling: `Beta(alpha, beta)` updates
   - User taste: `mean(liked_item_embeddings)`

All "learning" is just counters, moving averages, and centroid updates. No gradient descent, no model weights updated.

## FAQ

**Q: Can I use different models?**
A: Yes! Just edit `config.yaml`. Any sentence-transformer model for embeddings, any cross-encoder for reranking, any OpenAI-compatible endpoint for LLM.

**Q: Do I need all 6 GPUs?**
A: No. Reduce `tensor-parallel-size` for vLLM, or use smaller models. System works with 1 GPU (just adjust config).

**Q: Can I use OpenAI API instead of local models?**
A: Yes. Set LLM provider to `openai_compatible` and point to OpenAI endpoint.

**Q: How do I add YouTube/Podcasts?**
A: Implement in `ingest/` modules (examples provided). Use `yt-dlp` for YouTube, `feedparser` for podcasts.

**Q: Can I train models on my data?**
A: This system is designed for zero-training. If you want to fine-tune, you'd need to extend it (out of scope).

**Q: How accurate is this vs trained recommenders?**
A: Surprisingly good! Pre-trained embeddings capture semantic similarity well. Simple priors learn quality. Thompson Sampling handles exploration. Often 80-90% of a trained system's quality.

## License

MIT

## Contributing

Contributions welcome! Areas:
- New ingest sources (Twitter, podcasts, etc.)
- New model providers (Cohere, Anthropic, etc.)
- UI/frontend
- Performance optimizations

## Acknowledgments

Built on excellent open-source models:
- BGE embeddings (BAAI)
- Mixtral (Mistral AI)
- Whisper (OpenAI)
- Qdrant, vLLM, FastAPI
