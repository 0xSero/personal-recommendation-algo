"""Candidate generation using pure heuristics (no training)."""

from typing import List, Optional, Dict, Any, Set
import numpy as np
from collections import Counter
from datetime import datetime, timedelta

from ..database import VectorStore, DuckDBStore, Item, UserProfile, MediaType


class CandidateGenerator:
    """Generate candidate items using centroid-based similarity and metadata filters.

    NO TRAINING - just vector similarity + simple filters.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        db_store: DuckDBStore,
        config: Dict[str, Any],
    ):
        """Initialize candidate generator."""
        self.vector_store = vector_store
        self.db_store = db_store
        self.config = config

    def generate_candidates(
        self,
        user_profile: UserProfile,
        session_items: Optional[List[Item]] = None,
        num_candidates: int = 500,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate candidate items for recommendation.

        Args:
            user_profile: User's profile with preferences
            session_items: Recent items in current session
            num_candidates: Target number of candidates
            context: Optional context (time budget, format, etc.)

        Returns:
            List of candidate dicts with item_id and metadata
        """
        candidates = []
        seen_ids: Set[str] = set()

        # Get strategy configs
        strategies = self.config.get("strategies", [])

        for strategy in strategies:
            strategy_name = strategy["name"]
            top_k = strategy["top_k"]

            if strategy_name == "taste_centroid":
                # Similarity to global taste
                cands = self._taste_centroid_candidates(
                    user_profile, top_k, seen_ids, context
                )
            elif strategy_name == "session_context":
                # Similarity to recent session items
                cands = self._session_context_candidates(
                    user_profile, session_items, top_k, seen_ids, context
                )
            elif strategy_name == "topic_slots":
                # Pull from top topics
                cands = self._topic_slot_candidates(
                    user_profile, top_k, seen_ids, context
                )
            elif strategy_name == "exploration":
                # Random exploration from fresh/unseen
                cands = self._exploration_candidates(
                    user_profile, top_k, seen_ids, context
                )
            else:
                continue

            for cand in cands:
                candidates.append(
                    {
                        "item_id": cand.id,
                        "strategy": strategy_name,
                        "metadata": {
                            "source": cand.source,
                            "creator": cand.creator,
                            "media_type": cand.media_type.value,
                            "topics": cand.topics,
                            "length_seconds": cand.length_seconds,
                            "published_at": cand.published_at.isoformat()
                            if cand.published_at
                            else None,
                        },
                    }
                )
                seen_ids.add(cand.id)

        return candidates[:num_candidates]

    def _taste_centroid_candidates(
        self,
        user_profile: UserProfile,
        top_k: int,
        seen_ids: Set[str],
        context: Optional[Dict[str, Any]],
    ) -> List[Item]:
        """Get candidates similar to user's global taste vector."""
        if not user_profile.global_embedding:
            return []

        query_vector = np.array(user_profile.global_embedding)

        # Build filters from context
        filters = self._build_filters(user_profile, context)

        # Search
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k * 2,  # Get extra to account for filtering
            filters=filters,
        )

        # Convert to items
        items = []
        for result in results:
            if result.id in seen_ids:
                continue

            item = self.db_store.get_item(result.id)
            if item and not self._is_muted(item, user_profile):
                items.append(item)

            if len(items) >= top_k:
                break

        return items

    def _session_context_candidates(
        self,
        user_profile: UserProfile,
        session_items: Optional[List[Item]],
        top_k: int,
        seen_ids: Set[str],
        context: Optional[Dict[str, Any]],
    ) -> List[Item]:
        """Get candidates similar to recent session items."""
        if not session_items:
            return []

        # Compute session centroid (recent items)
        recent_n = self.config.get("recent_items", 5)
        recent = session_items[-recent_n:]

        # Get embeddings for recent items
        embeddings = []
        for item in recent:
            vec = self.vector_store.get_vector(item.id)
            if vec is not None:
                embeddings.append(vec)

        if not embeddings:
            return []

        # Weighted average (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, len(embeddings)))
        weights = weights / weights.sum()

        session_vector = np.average(embeddings, axis=0, weights=weights)

        # Build filters
        filters = self._build_filters(user_profile, context)

        # Search
        results = self.vector_store.search(
            query_vector=session_vector, top_k=top_k * 2, filters=filters
        )

        # Convert to items
        items = []
        for result in results:
            if result.id in seen_ids:
                continue

            item = self.db_store.get_item(result.id)
            if item and not self._is_muted(item, user_profile):
                items.append(item)

            if len(items) >= top_k:
                break

        return items

    def _topic_slot_candidates(
        self,
        user_profile: UserProfile,
        top_k: int,
        seen_ids: Set[str],
        context: Optional[Dict[str, Any]],
    ) -> List[Item]:
        """Get candidates from user's top topics."""
        if not user_profile.top_topics:
            return []

        num_topics = min(
            self.config.get("num_topics", 5), len(user_profile.top_topics)
        )
        top_topics = user_profile.top_topics[:num_topics]

        items = []
        per_topic = max(1, top_k // num_topics)

        for topic in top_topics:
            # Get items with this topic
            # For now, do a simple filter-based query
            # In production, you could maintain topic-specific indices
            topic_items = self._get_items_by_topic(
                topic, per_topic, seen_ids, user_profile, context
            )
            items.extend(topic_items)

            if len(items) >= top_k:
                break

        return items[:top_k]

    def _exploration_candidates(
        self,
        user_profile: UserProfile,
        top_k: int,
        seen_ids: Set[str],
        context: Optional[Dict[str, Any]],
    ) -> List[Item]:
        """Get random exploration candidates."""
        # Get fresh items (published recently)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)

        # Sample from low-seen creators
        # For now, just get recent items randomly
        # In production, you'd track creator view counts

        # Get random recent items
        all_items = self.db_store.get_items(limit=top_k * 5)

        # Filter and shuffle
        candidates = []
        for item in all_items:
            if item.id in seen_ids:
                continue
            if self._is_muted(item, user_profile):
                continue
            if item.published_at and item.published_at < recent_cutoff:
                continue

            candidates.append(item)

        # Shuffle
        np.random.shuffle(candidates)

        return candidates[:top_k]

    def _get_items_by_topic(
        self,
        topic: str,
        limit: int,
        seen_ids: Set[str],
        user_profile: UserProfile,
        context: Optional[Dict[str, Any]],
    ) -> List[Item]:
        """Get items for a specific topic."""
        # Simple implementation: get all items and filter by topic
        # In production, you'd maintain topic indices
        all_items = self.db_store.get_items(limit=limit * 5)

        topic_items = []
        for item in all_items:
            if item.id in seen_ids:
                continue
            if self._is_muted(item, user_profile):
                continue
            if topic.lower() in [t.lower() for t in item.topics]:
                topic_items.append(item)

            if len(topic_items) >= limit:
                break

        return topic_items

    def _build_filters(
        self, user_profile: UserProfile, context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Build metadata filters from user profile and context."""
        filters = {}

        # Format filters
        if context and context.get("preferred_format"):
            filters["media_type"] = context["preferred_format"]
        elif user_profile.preferred_formats:
            filters["media_type"] = [f.value for f in user_profile.preferred_formats]

        # Length filters (done at reranking, not here)

        return filters if filters else None

    def _is_muted(self, item: Item, user_profile: UserProfile) -> bool:
        """Check if item is muted by user."""
        if item.creator and item.creator in user_profile.muted_creators:
            return True

        for topic in item.topics:
            if topic in user_profile.muted_topics:
                return True

        return False
