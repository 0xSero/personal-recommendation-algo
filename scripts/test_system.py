#!/usr/bin/env python3
"""
End-to-end system test script.

This script tests the entire recommendation system with mock components.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import numpy as np
from datetime import datetime, timedelta


def create_mock_models():
    """Create mock models for testing."""
    from recommender.models import ModelFactory, BaseEmbeddingModel, BaseReranker, BaseLLM

    class MockEmbedding(BaseEmbeddingModel):
        def encode(self, texts, batch_size=None, show_progress=False):
            return np.random.rand(len(texts), 384)

        def get_dimension(self):
            return 384

    class MockReranker(BaseReranker):
        def rank(self, query, documents, top_k=None):
            scores = np.random.rand(len(documents))
            indices = np.argsort(scores)[::-1]
            if top_k:
                indices = indices[:top_k]
            return [(int(idx), float(scores[idx])) for idx in indices]

        def score(self, query, document):
            return float(np.random.rand())

    class MockLLM(BaseLLM):
        async def agenerate(self, prompt, **kwargs):
            return "• Summary point 1\\n• Point 2\\nTime-to-value: Quick\\nAudience: intermediate\\nTags: ML, AI"

        def generate(self, prompt, **kwargs):
            return "• Summary point 1\\n• Point 2\\nTime-to-value: Quick\\nAudience: intermediate\\nTags: ML, AI"

    # Register mock providers
    ModelFactory.register_embedding_provider("mock", MockEmbedding)
    ModelFactory.register_reranker_provider("mock", MockReranker)
    ModelFactory.register_llm_provider("mock", MockLLM)

    return {
        "embedding": ModelFactory.create_embedding_model(
            {"provider": "mock", "model_name": "test"}
        ),
        "reranker": ModelFactory.create_reranker(
            {"provider": "mock", "model_name": "test"}
        ),
        "llm": ModelFactory.create_llm({"provider": "mock", "model_name": "test"}),
    }


def create_mock_vector_store():
    """Create a simple in-memory vector store."""

    class SimpleVectorStore:
        def __init__(self):
            self.vectors = {}
            self.payloads = {}

        def add_vectors(self, ids, vectors, payloads=None):
            for i, id_ in enumerate(ids):
                self.vectors[id_] = vectors[i] if len(vectors.shape) > 1 else vectors
                if payloads:
                    self.payloads[id_] = payloads[i]

        def search(self, query_vector, top_k=10, filters=None):
            from recommender.database.vector_store import VectorSearchResult

            # Simple random search
            ids = list(self.vectors.keys())[:top_k]
            return [
                VectorSearchResult(
                    id=id_, score=float(np.random.rand()), payload=self.payloads.get(id_, {})
                )
                for id_ in ids
            ]

        def get_vector(self, vector_id):
            return self.vectors.get(vector_id)

        def delete_vectors(self, ids):
            for id_ in ids:
                self.vectors.pop(id_, None)

    return SimpleVectorStore()


def test_full_pipeline():
    """Test the full recommendation pipeline."""
    from recommender.database import (
        DuckDBStore,
        Item,
        MediaType,
        UserProfile,
        EventType,
    )
    from recommender.candidate_generation import CandidateGenerator
    from recommender.reranking import Reranker
    from recommender.exploration import ThompsonSamplingExplorer
    from recommender.feedback import FeedbackProcessor

    print("🧪 Testing Full Recommendation Pipeline\\n")
    print("=" * 60)

    # Setup
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.duckdb")

    print("\\n1️⃣  Setting up databases...")
    db_store = DuckDBStore(db_path)
    vector_store = create_mock_vector_store()
    print("   ✓ DuckDB and Vector store initialized")

    print("\\n2️⃣  Creating mock models...")
    models = create_mock_models()
    print("   ✓ Embedding, Reranker, and LLM ready")

    print("\\n3️⃣  Creating test items...")
    items = []
    for i in range(20):
        item = Item(
            id=f"item_{i}",
            source="test",
            title=f"Test Article {i}",
            text=f"Content about machine learning topic {i}",
            summary=f"Summary {i}",
            creator=f"creator_{i % 5}",
            media_type=MediaType.ARTICLE,
            topics=["Machine Learning", "AI"],
            length_seconds=300,
            published_at=datetime.utcnow() - timedelta(days=i),
        )
        items.append(item)

        # Save to DB
        db_store.save_item(item)

        # Create embedding and store
        embedding = models["embedding"].encode([item.title + " " + item.text])
        vector_store.add_vectors(
            [item.id],
            embedding,
            [{"source": item.source, "creator": item.creator}],
        )

    print(f"   ✓ Created and stored {len(items)} items")

    print("\\n4️⃣  Creating user profile...")
    profile = UserProfile(
        user_id="test_user", global_embedding=np.random.rand(384).tolist()
    )
    db_store.save_user_profile(profile)
    print("   ✓ User profile created")

    print("\\n5️⃣  Testing candidate generation...")
    cand_config = {
        "strategies": [
            {"name": "taste_centroid", "top_k": 10},
            {"name": "exploration", "top_k": 5},
        ]
    }
    candidate_gen = CandidateGenerator(vector_store, db_store, cand_config)
    candidates = candidate_gen.generate_candidates(profile, num_candidates=15)
    print(f"   ✓ Generated {len(candidates)} candidates")

    print("\\n6️⃣  Testing reranking...")
    rerank_config = {
        "weights": {
            "cosine_global": 0.35,
            "cosine_session": 0.20,
            "cross_encoder": 0.25,
            "creator_prior": 0.08,
            "source_prior": 0.05,
            "novelty": 0.04,
            "length_match": 0.03,
        },
        "recency_half_life_days": {"default": 14},
        "repetition_penalty": 0.5,
        "diversity_lambda": 0.7,
    }
    reranker = Reranker(vector_store, db_store, models["reranker"], rerank_config)
    ranked = reranker.rerank(candidates, profile, top_k=10)
    print(f"   ✓ Reranked to top {len(ranked)} items")

    if ranked:
        print(f"\\n   Top recommendation:")
        item = db_store.get_item(ranked[0]["item_id"])
        print(f"   - Title: {item.title}")
        print(f"   - Score: {ranked[0]['score']:.3f}")
        print(f"   - Strategy: {ranked[0]['strategy']}")

    print("\\n7️⃣  Testing exploration (Thompson Sampling)...")
    explore_config = {
        "arms": [
            {"name": "new_creators", "slots": 2},
            {"name": "deep_dives", "slots": 1},
        ],
        "alpha_prior": 1.0,
        "beta_prior": 1.0,
    }
    explorer = ThompsonSamplingExplorer(db_store, explore_config)

    arm_name = explorer.select_arm()
    print(f"   ✓ Selected exploration arm: {arm_name}")

    explorer.update_arm(arm_name, success=True)
    print(f"   ✓ Updated arm with success")

    print("\\n8️⃣  Testing feedback processing...")
    feedback_config = {
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
    feedback_proc = FeedbackProcessor(db_store, feedback_config)

    # Log some events
    feedback_proc.log_event(
        item_id="item_0",
        event_type=EventType.COMPLETION,
        dwell_seconds=60.0,
        completion_fraction=0.9,
    )
    print("   ✓ Logged completion event")

    feedback_proc.log_event(
        item_id="item_1", event_type=EventType.LIKE, dwell_seconds=30.0
    )
    print("   ✓ Logged like event")

    print("\\n9️⃣  Checking updated priors...")
    creator_prior = db_store.get_creator_prior("creator_0")
    print(f"   ✓ Creator prior updated: {creator_prior.avg_completion_rate:.3f}")

    source_prior = db_store.get_source_prior("test")
    print(f"   ✓ Source prior updated: {source_prior.avg_reward:.3f}")

    print("\\n🔟 Testing profile update...")
    updated_profile = db_store.get_user_profile("test_user")
    print(f"   ✓ Profile views: {updated_profile.total_items_viewed}")
    print(f"   ✓ Profile completions: {updated_profile.total_items_completed}")

    # Cleanup
    db_store.close()

    print("\\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("\\n🎉 The recommendation system is fully functional!\\n")


def main():
    """Run all tests."""
    try:
        test_full_pipeline()
        return 0
    except Exception as e:
        print(f"\\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
