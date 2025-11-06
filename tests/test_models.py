"""Tests for model interfaces and factory."""

import pytest
import numpy as np
from recommender.models import ModelFactory, ModelConfig


def test_model_config():
    """Test ModelConfig creation."""
    config = ModelConfig(
        provider="test",
        model_name="test-model",
        device="cpu",
        batch_size=16,
    )

    assert config.provider == "test"
    assert config.model_name == "test-model"
    assert config.device == "cpu"
    assert config.batch_size == 16


def test_embedding_factory():
    """Test embedding model factory."""
    config = {
        "provider": "sentence_transformers",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "batch_size": 32,
    }

    model = ModelFactory.create_embedding_model(config)

    # Test encoding
    texts = ["Hello world", "This is a test"]
    embeddings = model.encode(texts)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == model.get_dimension()
    assert isinstance(embeddings, np.ndarray)


def test_reranker_factory():
    """Test reranker factory."""
    config = {
        "provider": "sentence_transformers",
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "device": "cpu",
        "batch_size": 16,
    }

    model = ModelFactory.create_reranker(config)

    # Test ranking
    query = "Machine learning tutorial"
    documents = [
        "A comprehensive guide to machine learning",
        "Cooking recipes for beginners",
        "Introduction to deep learning",
    ]

    results = model.rank(query, documents, top_k=3)

    assert len(results) == 3
    assert all(isinstance(score, float) for _, score in results)
    # ML docs should rank higher
    assert results[0][0] in [0, 2]  # First or third doc


def test_model_factory_unknown_provider():
    """Test that unknown provider raises error."""
    config = {
        "provider": "unknown_provider",
        "model_name": "test",
    }

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        ModelFactory.create_embedding_model(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
