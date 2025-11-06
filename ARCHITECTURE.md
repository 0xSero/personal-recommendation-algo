# Architecture Deep Dive

Technical architecture of the model-agnostic, zero-training personal recommendation system.

## Core Principle: No Training

This system achieves personalization **without training any models**:

- ✅ Pre-trained embeddings (BGE, GTE, etc.)
- ✅ Pre-trained cross-encoders
- ✅ Pre-trained LLMs (Mixtral, Qwen, Llama)
- ✅ Simple counters and moving averages
- ✅ Thompson Sampling (Bayesian bandit)
- ❌ No gradient descent
- ❌ No fine-tuning
- ❌ No LightGBM/XGBoost
- ❌ No neural network training

## System Components

### 1. Model Layer (Fully Pluggable)

```
┌─────────────────────────────────────┐
│        Model Abstraction            │
├─────────────────────────────────────┤
│  BaseEmbeddingModel                 │
│  BaseReranker                       │
│  BaseLLM                            │
│  BaseASR                            │
└─────────────────────────────────────┘
          ▲
          │ implements
          │
┌─────────┴──────────────────────────┐
│  Concrete Implementations           │
├─────────────────────────────────────┤
│  - SentenceTransformerEmbedding     │
│  - SentenceTransformerReranker      │
│  - OpenAICompatibleLLM              │
│  - WhisperASR                       │
│  - FasterWhisperASR                 │
└─────────────────────────────────────┘
          ▲
          │ registered via
          │
┌─────────┴──────────────────────────┐
│      ModelFactory                   │
├─────────────────────────────────────┤
│  create_embedding_model(config)     │
│  create_reranker(config)            │
│  create_llm(config)                 │
│  create_asr(config)                 │
└─────────────────────────────────────┘
```

**Key Design:**
- All models use **abstract base classes**
- Swappable via **factory pattern**
- Config-driven (no code changes to swap models)
- Easy to add new providers

**Example: Adding Custom Provider**
```python
class MyCustomEmbedding(BaseEmbeddingModel):
    def encode(self, texts):
        # Your implementation
        return embeddings

    def get_dimension(self):
        return 768

# Register
ModelFactory.register_embedding_provider(
    "my_provider",
    MyCustomEmbedding
)

# Use in config.yaml
models:
  embedding:
    provider: "my_provider"
    model_name: "my-model"
```

### 2. Storage Layer

**Vector Storage (Qdrant or FAISS)**
```python
class VectorStore(ABC):
    def add_vectors(ids, vectors, payloads)
    def search(query_vector, top_k, filters)
    def get_vector(id)
    def delete_vectors(ids)
```

Stores:
- Item embeddings (1024d for BGE-large)
- Metadata for filtering (source, creator, media_type, topics)

**Relational Storage (DuckDB)**
```sql
-- Items table
CREATE TABLE items (
    id VARCHAR PRIMARY KEY,
    title VARCHAR,
    summary VARCHAR,  -- LLM-generated
    creator VARCHAR,
    topics VARCHAR,   -- JSON array
    length_seconds INTEGER,
    published_at TIMESTAMP,
    ...
);

-- Events table
CREATE TABLE events (
    id VARCHAR PRIMARY KEY,
    item_id VARCHAR,
    event_type VARCHAR,  -- click, completion, like, etc.
    timestamp TIMESTAMP,
    dwell_seconds DOUBLE,
    completion_fraction DOUBLE,
    ...
);

-- Priors (simple counters)
CREATE TABLE creator_priors (
    creator VARCHAR PRIMARY KEY,
    successes INTEGER,
    failures INTEGER,
    avg_completion_rate DOUBLE
);

CREATE TABLE source_priors (
    source VARCHAR PRIMARY KEY,
    avg_reward DOUBLE
);

-- Exploration arms (Thompson Sampling)
CREATE TABLE exploration_arms (
    name VARCHAR PRIMARY KEY,
    alpha DOUBLE,  -- successes + prior
    beta DOUBLE    -- failures + prior
);
```

Why DuckDB?
- **Fast analytics** on events
- **OLAP-style queries** for aggregations
- **Embedded** (no separate server)
- **Columnar** storage for efficiency

### 3. Recommendation Pipeline

#### Phase 1: Candidate Generation

**Goal:** Retrieve ~500 items quickly from millions

```python
def generate_candidates(user_profile, session_items, num_candidates=500):
    candidates = []

    # Strategy 1: Taste Centroid (40% weight)
    # Query: user's global taste vector
    u_global = mean(embeddings of liked items)
    results = vector_store.search(u_global, top_k=200)
    candidates.extend(results)

    # Strategy 2: Session Context (30% weight)
    # Query: weighted average of recent session items
    u_session = weighted_mean(session_items, recency_weights)
    results = vector_store.search(u_session, top_k=150)
    candidates.extend(results)

    # Strategy 3: Topic Slots (20% weight)
    # Pull from user's top 5 topics
    for topic in user_profile.top_topics[:5]:
        results = get_items_by_topic(topic, top_k=20)
        candidates.extend(results)

    # Strategy 4: Exploration (10% weight)
    # Random sample from fresh/unseen items
    results = sample_fresh_items(top_k=50)
    candidates.extend(results)

    return candidates[:num_candidates]
```

**Metadata Filters:**
```python
filters = {
    "media_type": ["article", "video"],
    "source": ["rss", "youtube"]
}
results = vector_store.search(query, filters=filters)
```

#### Phase 2: Reranking

**Goal:** Precisely score 500 candidates

```python
def rerank(candidates, user_profile, session_items, context):
    scores = {}

    for item in candidates:
        # 1. Cosine similarity to global taste
        cos_global = cosine(user_profile.global_embedding, item.embedding)

        # 2. Cosine similarity to session
        cos_session = cosine(session_embedding, item.embedding)

        # 3. Cross-encoder score (expensive but accurate)
        query = build_query(user_profile, session_items, context)
        cross_score = cross_encoder.score(query, item.title + item.summary)

        # 4. Creator prior (simple counter-based)
        creator_prior = db.get_creator_prior(item.creator)
        creator_score = creator_prior.avg_completion_rate

        # 5. Source prior (simple EMA)
        source_prior = db.get_source_prior(item.source)
        source_score = source_prior.avg_reward

        # 6. Novelty (inverse frequency)
        novelty = 1.0 / (1.0 + recent_creator_counts[item.creator])

        # 7. Length match (Gaussian)
        if context.time_budget:
            length_match = exp(-(item.length - context.time_budget)^2 / 200)

        # 8. Recency decay (exponential)
        age_days = (now - item.published_at).days
        half_life = recency_half_life[item.source]
        recency = exp(-log(2) * age_days / half_life)

        # Weighted sum
        total_score = (
            0.35 * cos_global +
            0.20 * cos_session +
            0.25 * cross_score +
            0.08 * creator_score +
            0.05 * source_score +
            0.04 * novelty +
            0.03 * length_match +
            recency
        )

        scores[item.id] = total_score

    # Sort by score
    ranked = sorted(candidates, key=lambda x: scores[x.id], reverse=True)

    # Apply MMR for diversity
    diversified = apply_mmr(ranked, scores, lambda_=0.7)

    return diversified[:top_k]
```

**MMR (Maximal Marginal Relevance):**
```python
def apply_mmr(items, scores, lambda_=0.7):
    selected = []
    remaining = items.copy()

    while len(selected) < top_k and remaining:
        if not selected:
            # First item: pure relevance
            best = max(remaining, key=lambda x: scores[x.id])
        else:
            # Subsequent: balance relevance vs diversity
            best_mmr = -inf
            best_item = None

            for item in remaining:
                relevance = scores[item.id]

                # Max similarity to already selected
                max_sim = max(
                    cosine(item.embedding, sel.embedding)
                    for sel in selected
                )

                # MMR = λ*relevance - (1-λ)*similarity
                mmr = lambda_ * relevance - (1 - lambda_) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_item = item

            best = best_item

        selected.append(best)
        remaining.remove(best)

    return selected
```

#### Phase 3: Exploration

**Thompson Sampling** (Bayesian multi-armed bandit):

```python
class ExplorationArm:
    name: str
    alpha: float  # successes + prior
    beta: float   # failures + prior

    def sample(self):
        # Sample from Beta distribution
        return np.random.beta(self.alpha, self.beta)

    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1

# Select arm
arms = [
    ExplorationArm("new_creators", alpha=1, beta=1),
    ExplorationArm("deep_dives", alpha=1, beta=1),
    ExplorationArm("short_sharp", alpha=1, beta=1),
    ExplorationArm("hot_news", alpha=1, beta=1)
]

samples = [(arm, arm.sample()) for arm in arms]
selected_arm = max(samples, key=lambda x: x[1])[0]

# Get items for this arm
if selected_arm.name == "new_creators":
    items = get_unseen_creators()
elif selected_arm.name == "deep_dives":
    items = get_long_form_content()
# ...
```

Why Thompson Sampling?
- **No model needed** - just Beta distributions
- **Automatically balances** exploration vs exploitation
- **Converges quickly** to good strategies
- **Bayesian** - principled uncertainty quantification

### 4. Feedback Loop

**Reward Computation:**
```python
def compute_reward(event):
    reward = 0.0

    # Completion component (60%)
    reward += 0.6 * event.completion_fraction

    # Dwell component (20%)
    dwell_score = min(event.dwell_seconds / 120, 1.0)
    reward += 0.2 * dwell_score

    # Like component (20%)
    if event.type == "like":
        reward += 0.2

    # Bounce penalty (-30%)
    if event.type == "hide" or event.dwell_seconds < 10:
        reward -= 0.3

    return clip(reward, 0, 1)
```

**Prior Updates (Exponential Moving Average):**
```python
class CreatorPrior:
    successes: int
    failures: int
    avg_completion_rate: float

    def update(self, success: bool, decay=0.95):
        if success:
            self.successes += 1
        else:
            self.failures += 1

        # EMA update
        new_rate = self.successes / max(self.total_views, 1)
        self.avg_completion_rate = (
            decay * self.avg_completion_rate +
            (1 - decay) * new_rate
        )
```

Why EMAs?
- **No model** - just arithmetic
- **Adaptive** - recent data matters more
- **Stable** - smooths out noise
- **Fast** - O(1) update

**User Taste Update:**
```python
# Simple centroid update
def update_user_taste(user_profile, liked_item):
    if not user_profile.global_embedding:
        user_profile.global_embedding = liked_item.embedding
    else:
        # Moving average
        user_profile.global_embedding = (
            0.95 * user_profile.global_embedding +
            0.05 * liked_item.embedding
        )
```

### 5. Content Pipeline

```
Sources → Ingest → Enrich → Vectorize → Store
```

**Ingest:**
```python
class RSSIngester(BaseIngester):
    async def fetch_items(self):
        feeds = read_feeds_file()
        items = []

        for feed_url in feeds:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                item = Item(
                    id=hash(entry.link),
                    source="rss",
                    url=entry.link,
                    title=entry.title,
                    text=entry.summary,
                    published_at=parse_date(entry)
                )
                items.append(item)

        return items
```

**Enrich (LLM Zero-Shot):**
```python
async def enrich_item(item):
    prompt = f"""
    Summarize this content in 5 bullet points.
    Then add:
    - Time-to-value: (1 sentence)
    - Audience: (novice/intermediate/advanced)
    - Tags: (3 from: {TOPICS})

    Title: {item.title}
    Text: {item.text[:2000]}
    """

    response = await llm.generate(prompt)

    # Parse response
    item.summary = extract_summary(response)
    item.topics = extract_tags(response)
    item.audience_level = extract_audience(response)

    return item
```

**Vectorize:**
```python
# Batch encode
texts = [f"{item.title}. {item.summary}" for item in items]
embeddings = embedding_model.encode(texts, batch_size=32)

# Store
for item, emb in zip(items, embeddings):
    vector_store.add_vectors([item.id], emb, [{
        "source": item.source,
        "creator": item.creator,
        "topics": item.topics
    }])
```

## Data Flow

### Request → Response

```
1. Request arrives
   GET /recommend?num_items=20&time_budget=30

2. Load user profile from DuckDB
   user = db.get_user_profile()

3. Generate candidates (500 items)
   candidates = candidate_generator.generate(user, num=500)
   # Uses vector search + metadata filters

4. Rerank (top 50)
   ranked = reranker.rerank(candidates, user, top_k=50)
   # Computes weighted scores + MMR

5. Add exploration (5 items)
   exploration = explorer.sample_arms(num=5)
   results = ranked[:15] + exploration

6. Return
   return {"items": results, "session_id": session.id}
```

### Feedback → Learning

```
1. Feedback arrives
   POST /feedback {item_id, event_type, dwell, completion}

2. Compute reward
   reward = 0.6*completion + 0.2*dwell + 0.2*like - 0.3*bounce

3. Update creator prior
   creator_prior = db.get_creator_prior(item.creator)
   creator_prior.update(success=reward > 0.5)
   db.save(creator_prior)

4. Update source prior
   source_prior = db.get_source_prior(item.source)
   source_prior.update(reward)
   db.save(source_prior)

5. Update exploration arm
   arm = db.get_arm(item.arm_name)
   arm.update(success=reward > 0.7)
   db.save(arm)

6. Update user taste (if high reward)
   if reward > 0.7:
       user.global_embedding = 0.95*user.global_embedding + 0.05*item.embedding
       user.top_topics.add(item.topics)
       db.save(user)
```

## Performance Characteristics

### Latency

| Component | Latency | Notes |
|-----------|---------|-------|
| Vector search (Qdrant) | ~20-50ms | For 1M items |
| Candidate generation | ~50ms | Parallel searches |
| Cross-encoder (500 items) | ~200ms | Batched on GPU |
| Reranking (total) | ~250ms | Including MMR |
| Full request | ~300-500ms | End-to-end |

### Throughput

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Embeddings | ~1000 items/min | Batched, GPU |
| Reranking | ~100 items/sec | GPU cross-encoder |
| Recommendations | ~20 req/sec | With 1 API worker |
| Ingestion | ~500 items/hour | LLM enrichment bottleneck |

### Memory

| Component | VRAM | RAM |
|-----------|------|-----|
| Mixtral-8×7B (vLLM) | ~80GB | ~10GB |
| BGE-large embedding | ~4GB | ~2GB |
| BGE-reranker-large | ~4GB | ~2GB |
| Qdrant | - | ~1GB/1M items |
| API server | - | ~2GB |

## Scalability

### Vertical Scaling

- **More GPUs** → increase tensor parallelism for LLM
- **Bigger GPUs** → use larger models (Llama-70B, etc.)
- **More RAM** → cache more embeddings in memory

### Horizontal Scaling

- **Multiple API workers** → load balance with nginx
- **Separate embedding service** → dedicated GPU for embeddings
- **Qdrant cluster** → shard vector DB across nodes
- **DuckDB → Postgres** → for multi-user scenarios

### Optimizations

**1. Caching:**
```python
# Cache embeddings
@lru_cache(maxsize=10000)
def get_embedding(item_id):
    return vector_store.get_vector(item_id)

# Cache cross-encoder scores (session-level)
session_cache[query_hash][item_id] = cross_score
```

**2. Batching:**
```python
# Batch cross-encoder inference
pairs = [(query, doc) for doc in documents]
scores = cross_encoder.predict(pairs, batch_size=64)
```

**3. Quantization:**
```yaml
# vLLM with AWQ/GPTQ
command:
  - --quantization
  - awq  # or gptq
```

**4. Pruning:**
```python
# Filter before reranking
candidates = [c for c in candidates if c.score > threshold]
```

## Extension Points

### Adding New Models

Implement interface → register → configure:

```python
# 1. Implement
class CoherereRanker(BaseReranker):
    def rank(self, query, docs):
        # Call Cohere API
        ...

# 2. Register
ModelFactory.register_reranker_provider("cohere", CohereRanker)

# 3. Configure
# config.yaml
models:
  reranker:
    provider: "cohere"
    api_key: "..."
```

### Adding New Sources

```python
class TwitterIngester(BaseIngester):
    async def fetch_items(self):
        # Fetch tweets
        ...
        return items

# Use in ingestion pipeline
ingesters.append(TwitterIngester(config))
```

### Adding New Scoring Signals

```python
# In reranker.py
def _compute_scores(self, items, user_profile):
    # ... existing signals ...

    # Add new signal
    components["my_signal"] = compute_my_signal(item)

    # Add to total
    total_score += self.weight_my_signal * components["my_signal"]
```

### Adding New Exploration Strategies

```python
# In exploration/thompson_sampling.py
def _get_item_for_arm(self, arm_name):
    if arm_name == "my_new_strategy":
        return self._my_strategy_items()
```

## Why This Architecture Works

1. **Model Agnostic**: Swap models without touching code
2. **Zero Training**: All models pre-trained, learning via counters
3. **Fast**: Vector search + simple arithmetic
4. **Scalable**: Horizontal and vertical
5. **Extensible**: Clear interfaces for adding features
6. **Debuggable**: Simple logic, no black boxes
7. **Resource Efficient**: Local, no API costs

## Trade-offs vs Trained Systems

### Advantages
- ✅ **No training data needed** - works day 1
- ✅ **No training infrastructure** - just inference
- ✅ **Fully transparent** - no black box
- ✅ **Easy to debug** - simple logic
- ✅ **Fast iteration** - change config, done

### Limitations
- ❌ Not as accurate as trained rankers (80-90% of performance)
- ❌ Cross-encoder reranking is slower than learned models
- ❌ Simple priors don't capture complex interactions
- ❌ Can't learn from implicit signals as well

### When to Use This Architecture

**Good for:**
- Personal use (1 user)
- Prototyping recommendation systems
- Privacy-sensitive applications (all local)
- Domains with little training data
- When interpretability matters

**Not good for:**
- Massive scale (YouTube, Netflix size)
- When 99th percentile latency critical
- When you have millions of training labels

## Conclusion

This architecture proves that **pre-trained models + simple heuristics** can build a surprisingly effective recommendation system **without any training**. The key is:

1. Use strong pre-trained models (BGE, Mixtral)
2. Simple but principled scoring (weighted sum, MMR)
3. Lightweight learning (counters, EMAs, Thompson Sampling)
4. Model agnostic design (easy to upgrade as models improve)

As foundation models get better, this approach gets better for free.
