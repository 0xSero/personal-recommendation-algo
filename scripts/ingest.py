#!/usr/bin/env python3
"""Script to run ingestion pipeline."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from recommender.models import ModelFactory
from recommender.database import DuckDBStore, QdrantVectorStore
from recommender.ingest import RSSIngester
from recommender.enrichment import ContentEnricher


async def main():
    """Run ingestion pipeline."""
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    print("🚀 Starting ingestion pipeline...")

    # Initialize models
    print("📦 Loading models...")
    embedding_model = ModelFactory.create_embedding_model(config["models"]["embedding"])
    llm = ModelFactory.create_llm(config["models"]["llm"])

    # Initialize databases
    print("💾 Connecting to databases...")
    db_store = DuckDBStore(config["relational_db"]["path"])
    vector_store = QdrantVectorStore(
        host=config["vector_db"]["host"],
        port=config["vector_db"]["port"],
        collection_name=config["vector_db"]["collection_name"],
        dimension=embedding_model.get_dimension(),
    )

    # Initialize enricher
    enricher = ContentEnricher(llm=llm, config=config["enrichment"])

    # Initialize ingesters
    ingesters = []

    if config["sources"]["rss"]["enabled"]:
        print("📰 Initializing RSS ingester...")
        ingesters.append(RSSIngester(config["sources"]["rss"]))

    # Fetch items from all sources
    print("🔍 Fetching items...")
    all_items = []
    for ingester in ingesters:
        try:
            items = await ingester.fetch_items()
            print(f"  ✓ {ingester.get_source_name()}: {len(items)} items")
            all_items.extend(items)
        except Exception as e:
            print(f"  ✗ {ingester.get_source_name()}: {e}")

    if not all_items:
        print("No items fetched. Exiting.")
        return

    # Enrich items (LLM summaries)
    print(f"\n🤖 Enriching {len(all_items)} items with LLM...")
    for i, item in enumerate(all_items):
        try:
            item = await enricher.enrich_item(item)
            print(f"  [{i+1}/{len(all_items)}] {item.title[:60]}...")
        except Exception as e:
            print(f"  ✗ Error enriching item: {e}")

    # Generate embeddings
    print("\n🔢 Generating embeddings...")
    texts = []
    for item in all_items:
        # Combine title + summary for embedding
        text = f"{item.title}. {item.summary}"
        texts.append(text)

    embeddings = embedding_model.encode(texts, show_progress=True)

    # Store in databases
    print("\n💾 Storing items...")
    for item, embedding in zip(all_items, embeddings):
        # Save to DuckDB
        db_store.save_item(item)

        # Save to Qdrant
        vector_store.add_vectors(
            ids=[item.id],
            vectors=embedding.reshape(1, -1),
            payloads=[
                {
                    "source": item.source,
                    "creator": item.creator,
                    "media_type": item.media_type.value,
                    "topics": item.topics,
                }
            ],
        )

    print(f"\n✅ Ingested {len(all_items)} items successfully!")


if __name__ == "__main__":
    asyncio.run(main())
