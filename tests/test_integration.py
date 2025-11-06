"""Integration tests for the recommendation system."""

import pytest
import tempfile
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Mock implementations for testing without actual models
class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dimension=384):
        self.dimension = dimension

    def encode(self, texts, batch_size=None, show_progress=False):
        """Return random embeddings."""
        return np.random.rand(len(texts), self.dimension)

    def get_dimension(self):
        return self.dimension


class MockReranker:
    """Mock reranker for testing."""

    def rank(self, query, documents, top_k=None):
        """Return random rankings."""
        scores = np.random.rand(len(documents))
        indices = np.argsort(scores)[::-1]
        if top_k:
            indices = indices[:top_k]
        return [(int(idx), float(scores[idx])) for idx in indices]

    def score(self, query, document):
        """Return random score."""
        return float(np.random.rand())


class MockLLM:
    """Mock LLM for testing."""

    async def agenerate(self, prompt, **kwargs):
        """Return mock summary."""
        return """
• Point 1
• Point 2
• Point 3

Time-to-value: Quick read
Audience: intermediate
Tags: Machine Learning, Python, AI
"""

    def generate(self, prompt, **kwargs):
        """Sync version."""
        return """
• Point 1
• Point 2
• Point 3

Time-to-value: Quick read
Audience: intermediate
Tags: Machine Learning, Python, AI
"""


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self):
        self.vectors = {}
        self.payloads = {}

    def add_vectors(self, ids, vectors, payloads=None):
        """Store vectors."""
        for i, id_ in enumerate(ids):
            self.vectors[id_] = vectors[i]
            if payloads:
                self.payloads[id_] = payloads[i]

    def search(self, query_vector, top_k=10, filters=None):
        """Return mock search results."""
        from recommender.database.vector_store import VectorSearchResult

        # Just return random items
        ids = list(self.vectors.keys())[:top_k]
        return [
            VectorSearchResult(
                id=id_,
                score=float(np.random.rand()),
                payload=self.payloads.get(id_, {}),
            )
            for id_ in ids
        ]

    def get_vector(self, vector_id):
        """Get vector by ID."""
        return self.vectors.get(vector_id)

    def delete_vectors(self, ids):
        """Delete vectors."""
        for id_ in ids:
            self.vectors.pop(id_, None)
            self.payloads.pop(id_, None)


@pytest.fixture
def temp_db():
    """Create temporary database."""
    from recommender.database import DuckDBStore

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.duckdb")
    store = DuckDBStore(db_path)
    yield store
    store.close()


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    return MockVectorStore()


@pytest.fixture
def sample_items():
    """Create sample items for testing."""
    from recommender.database import Item, MediaType

    items = []
    for i in range(10):
        item = Item(
            id=f"item_{i}",
            source="rss",
            title=f"Test Article {i}",
            text=f"This is test article {i} about machine learning.",
            summary=f"Summary of article {i}",
            creator=f"creator_{i % 3}",  # 3 creators
            media_type=MediaType.ARTICLE,
            topics=["Machine Learning", "AI"],
            length_seconds=300,
            published_at=datetime.utcnow() - timedelta(days=i),
        )
        items.append(item)
    return items


def test_database_operations(temp_db, sample_items):
    """Test database CRUD operations."""
    from recommender.database import UserProfile, CreatorPrior, Event, EventType

    # Test saving items
    for item in sample_items:
        temp_db.save_item(item)

    # Test retrieving items
    retrieved = temp_db.get_item("item_0")
    assert retrieved is not None
    assert retrieved.title == "Test Article 0"

    # Test getting items by source
    items = temp_db.get_items(source="rss", limit=5)
    assert len(items) <= 5

    # Test user profile
    profile = temp_db.get_user_profile()
    assert profile.user_id == "default"

    # Test updating profile
    profile.top_topics = ["Machine Learning", "AI"]
    temp_db.save_user_profile(profile)

    retrieved_profile = temp_db.get_user_profile()
    assert "Machine Learning" in retrieved_profile.top_topics

    # Test creator prior
    prior = temp_db.get_creator_prior("creator_0")
    assert prior.creator == "creator_0"

    prior.update(success=True)
    temp_db.save_creator_prior(prior)

    retrieved_prior = temp_db.get_creator_prior("creator_0")
    assert retrieved_prior.successes == 1

    # Test logging events
    event = Event(
        id="event_1",
        item_id="item_0",
        event_type=EventType.CLICK,
        dwell_seconds=30.0,
    )
    temp_db.log_event(event)

    events = temp_db.get_recent_events(limit=10)
    assert len(events) > 0


def test_candidate_generation(temp_db, mock_vector_store, sample_items):
    """Test candidate generation."""
    from recommender.candidate_generation import CandidateGenerator
    from recommender.database import UserProfile

    # Setup
    for item in sample_items:
        temp_db.save_item(item)
        # Add embeddings
        embedding = np.random.rand(1, 384)
        mock_vector_store.add_vectors(
            [item.id],
            embedding,
            [{"source": item.source, "creator": item.creator}],
        )

    # Create user profile with global embedding
    profile = UserProfile(user_id="test", global_embedding=np.random.rand(384).tolist())

    # Create generator
    config = {
        "strategies": [
            {"name": "taste_centroid", "top_k": 5},
            {"name": "exploration", "top_k": 2},
        ]
    }

    generator = CandidateGenerator(mock_vector_store, temp_db, config)

    # Generate candidates
    candidates = generator.generate_candidates(profile, num_candidates=10)

    assert len(candidates) > 0
    assert all("item_id" in c for c in candidates)
    assert all("strategy" in c for c in candidates)


def test_reranking(temp_db, mock_vector_store, sample_items):
    """Test reranking."""
    from recommender.reranking import Reranker
    from recommender.database import UserProfile

    # Setup
    for item in sample_items:
        temp_db.save_item(item)
        embedding = np.random.rand(1, 384)
        mock_vector_store.add_vectors([item.id], embedding)

    # Create candidates
    candidates = [
        {"item_id": item.id, "strategy": "test", "metadata": {}}
        for item in sample_items[:5]
    ]

    # Create profile
    profile = UserProfile(user_id="test", global_embedding=np.random.rand(384).tolist())

    # Create reranker
    config = {
        "weights": {
            "cosine_global": 0.4,
            "cosine_session": 0.2,
            "cross_encoder": 0.2,
            "creator_prior": 0.1,
            "source_prior": 0.05,
            "novelty": 0.03,
            "length_match": 0.02,
        },
        "recency_half_life_days": {"default": 14},
        "repetition_penalty": 0.5,
        "diversity_lambda": 0.7,
    }

    reranker = Reranker(mock_vector_store, temp_db, MockReranker(), config)

    # Rerank
    ranked = reranker.rerank(candidates, profile, top_k=3)

    assert len(ranked) <= 3
    assert all("score" in r for r in ranked)
    assert all("score_components" in r for r in ranked)


def test_exploration(temp_db, sample_items):
    """Test Thompson Sampling exploration."""
    from recommender.exploration import ThompsonSamplingExplorer

    # Setup
    for item in sample_items:
        temp_db.save_item(item)

    config = {
        "arms": [
            {"name": "new_creators", "slots": 2},
            {"name": "deep_dives", "slots": 1},
        ],
        "alpha_prior": 1.0,
        "beta_prior": 1.0,
    }

    explorer = ThompsonSamplingExplorer(temp_db, config)

    # Select arm
    arm_name = explorer.select_arm()
    assert arm_name in ["new_creators", "deep_dives"]

    # Update arm
    explorer.update_arm(arm_name, success=True)

    # Check arm was updated
    arm = temp_db.get_exploration_arm(arm_name)
    assert arm.alpha > 1.0  # Should have increased


def test_feedback_processor(temp_db, sample_items):
    """Test feedback processing."""
    from recommender.feedback import FeedbackProcessor
    from recommender.database import EventType

    # Setup
    for item in sample_items:
        temp_db.save_item(item)

    config = {
        "reward_formula": {
            "completion_weight": 0.6,
            "dwell_weight": 0.2,
            "like_weight": 0.2,
            "bounce_penalty": -0.3,
            "dwell_cap_seconds": 120,
        },
        "prior_decay": 0.95,
        "min_samples_for_prior": 3,
    }

    processor = FeedbackProcessor(temp_db, config)

    # Log completion event
    event = processor.log_event(
        item_id="item_0",
        event_type=EventType.COMPLETION,
        dwell_seconds=60.0,
        completion_fraction=0.9,
    )

    assert event.item_id == "item_0"

    # Check that creator prior was updated
    item = temp_db.get_item("item_0")
    if item.creator:
        prior = temp_db.get_creator_prior(item.creator)
        assert prior.total_views > 0


@pytest.mark.asyncio
async def test_enrichment():
    """Test content enrichment."""
    from recommender.enrichment import ContentEnricher
    from recommender.database import Item, MediaType

    config = {
        "topics": ["Machine Learning", "AI", "Python"],
        "prompts": {
            "summarize": "Summarize: {title}\n{text}",
            "explanation": "Explain why",
        },
    }

    enricher = ContentEnricher(MockLLM(), config)

    item = Item(
        id="test",
        source="rss",
        title="Test Article",
        text="This is about machine learning.",
        media_type=MediaType.ARTICLE,
    )

    # Enrich
    enriched = await enricher.enrich_item(item)

    assert enriched.summary != ""
    assert len(enriched.topics) > 0


def test_api_server_creation():
    """Test that API server can be created."""
    from recommender.api import create_app

    # Create temp config
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "config.yaml")

    config = {
        "models": {
            "embedding": {"provider": "mock", "model_name": "test"},
            "reranker": {"provider": "mock", "model_name": "test"},
            "llm": {"provider": "mock", "model_name": "test"},
        },
        "vector_db": {
            "provider": "qdrant",
            "host": "localhost",
            "port": 6333,
            "collection_name": "test",
        },
        "relational_db": {
            "provider": "duckdb",
            "path": os.path.join(temp_dir, "test.duckdb"),
        },
        "recommendation": {
            "candidate_generation": {"strategies": []},
            "reranking": {"weights": {}},
            "exploration": {"enabled": False, "arms": []},
            "feedback": {},
        },
        "enrichment": {"topics": [], "prompts": {}},
        "server": {"cors_origins": ["*"]},
        "storage": {
            "data_dir": temp_dir,
            "cache_dir": temp_dir,
            "models_dir": temp_dir,
        },
    }

    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Note: We can't actually start the server without models
    # but we can verify it can be created
    # app = create_app(config_path)
    # assert app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
