"""Reranking with weighted scoring (NO TRAINING - just heuristics)."""

from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from ..database import (
    VectorStore,
    DuckDBStore,
    Item,
    UserProfile,
    CreatorPrior,
    SourcePrior,
)
from ..models import BaseReranker


class Reranker:
    """Rerank candidates using configurable weighted scoring.

    NO TRAINING - just computes scores from:
    - Vector similarity (global + session)
    - Cross-encoder scores
    - Simple priors (creator, source)
    - Recency decay
    - Novelty bonus
    - Length match
    - Diversity (MMR)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        db_store: DuckDBStore,
        cross_encoder: Optional[BaseReranker] = None,
        config: Dict[str, Any] = None,
    ):
        """Initialize reranker."""
        self.vector_store = vector_store
        self.db_store = db_store
        self.cross_encoder = cross_encoder
        self.config = config or {}

        # Get weights from config
        weights = self.config.get("weights", {})
        self.weight_cosine_global = weights.get("cosine_global", 0.35)
        self.weight_cosine_session = weights.get("cosine_session", 0.20)
        self.weight_cross_encoder = weights.get("cross_encoder", 0.25)
        self.weight_creator_prior = weights.get("creator_prior", 0.08)
        self.weight_source_prior = weights.get("source_prior", 0.05)
        self.weight_novelty = weights.get("novelty", 0.04)
        self.weight_length_match = weights.get("length_match", 0.03)

        # Decay parameters
        self.recency_half_life = self.config.get("recency_half_life_days", {})
        self.repetition_penalty = self.config.get("repetition_penalty", 0.5)
        self.diversity_lambda = self.config.get("diversity_lambda", 0.7)

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: UserProfile,
        session_items: Optional[List[Item]] = None,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates.

        Args:
            candidates: List of candidate dicts from candidate generation
            user_profile: User profile
            session_items: Recent session items
            context: Optional context (time budget, etc.)
            top_k: Number of items to return

        Returns:
            Reranked list of candidate dicts with scores
        """
        if not candidates:
            return []

        # Load full items
        items = []
        for cand in candidates:
            item = self.db_store.get_item(cand["item_id"])
            if item:
                items.append((item, cand))

        if not items:
            return []

        # Compute all scores
        scores = self._compute_scores(items, user_profile, session_items, context)

        # Apply MMR for diversity
        ranked_items = self._apply_mmr(
            items, scores, user_profile, top_k, self.diversity_lambda
        )

        # Format results
        results = []
        for item, cand, score, components in ranked_items:
            results.append(
                {
                    **cand,
                    "score": score,
                    "score_components": components,
                }
            )

        return results

    def _compute_scores(
        self,
        items: List[tuple],
        user_profile: UserProfile,
        session_items: Optional[List[Item]],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute all scoring components for items."""
        scores = {}

        # Get user global embedding
        global_emb = (
            np.array(user_profile.global_embedding)
            if user_profile.global_embedding
            else None
        )

        # Get session embedding
        session_emb = self._compute_session_embedding(session_items)

        # Get time budget from context
        time_budget_minutes = (
            context.get("time_budget_minutes") if context else None
        )

        # Track seen creators for novelty
        recent_creators = self._get_recent_creators(user_profile, session_items)

        # Prepare cross-encoder inputs if available
        if self.cross_encoder:
            cross_encoder_query = self._build_cross_encoder_query(
                user_profile, session_items, context
            )
            item_texts = [
                f"{item.title}. {item.summary}" for item, _ in items
            ]
            # Batch score
            cross_encoder_scores = {}
            rankings = self.cross_encoder.rank(cross_encoder_query, item_texts)
            for idx, score in rankings:
                cross_encoder_scores[idx] = score
        else:
            cross_encoder_scores = {}

        # Score each item
        for idx, (item, cand) in enumerate(items):
            components = {}

            # 1. Cosine similarity to global taste
            if global_emb is not None:
                item_emb = self.vector_store.get_vector(item.id)
                if item_emb is not None:
                    components["cosine_global"] = float(
                        np.dot(global_emb, item_emb)
                        / (np.linalg.norm(global_emb) * np.linalg.norm(item_emb))
                    )
                else:
                    components["cosine_global"] = 0.0
            else:
                components["cosine_global"] = 0.0

            # 2. Cosine similarity to session
            if session_emb is not None:
                item_emb = self.vector_store.get_vector(item.id)
                if item_emb is not None:
                    components["cosine_session"] = float(
                        np.dot(session_emb, item_emb)
                        / (np.linalg.norm(session_emb) * np.linalg.norm(item_emb))
                    )
                else:
                    components["cosine_session"] = 0.0
            else:
                components["cosine_session"] = 0.0

            # 3. Cross-encoder score
            components["cross_encoder"] = cross_encoder_scores.get(idx, 0.5)

            # 4. Creator prior
            if item.creator:
                creator_prior = self.db_store.get_creator_prior(item.creator)
                components["creator_prior"] = creator_prior.avg_completion_rate
            else:
                components["creator_prior"] = 0.5

            # 5. Source prior
            source_prior = self.db_store.get_source_prior(item.source)
            components["source_prior"] = source_prior.avg_reward

            # 6. Novelty (lower if seen this creator recently)
            creator_count = recent_creators.get(item.creator, 0)
            components["novelty"] = 1.0 / (1.0 + creator_count)

            # 7. Length match
            if time_budget_minutes and item.length_seconds:
                item_minutes = item.length_seconds / 60
                # Gaussian-like match
                diff = abs(item_minutes - time_budget_minutes)
                components["length_match"] = np.exp(-(diff**2) / (2 * 10**2))
            else:
                components["length_match"] = 0.5

            # 8. Recency decay
            components["recency_decay"] = self._compute_recency_decay(item)

            # 9. Repetition penalty (if seen very recently)
            components["repetition_penalty"] = self._compute_repetition_penalty(
                item, user_profile, session_items
            )

            # Compute total score
            total_score = (
                self.weight_cosine_global * components["cosine_global"]
                + self.weight_cosine_session * components["cosine_session"]
                + self.weight_cross_encoder * components["cross_encoder"]
                + self.weight_creator_prior * components["creator_prior"]
                + self.weight_source_prior * components["source_prior"]
                + self.weight_novelty * components["novelty"]
                + self.weight_length_match * components["length_match"]
                + components["recency_decay"]
                - components["repetition_penalty"]
            )

            scores[item.id] = {
                "total": total_score,
                "components": components,
            }

        return scores

    def _compute_session_embedding(
        self, session_items: Optional[List[Item]]
    ) -> Optional[np.ndarray]:
        """Compute weighted average of session item embeddings."""
        if not session_items:
            return None

        embeddings = []
        for item in session_items[-5:]:  # Last 5 items
            emb = self.vector_store.get_vector(item.id)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return None

        # Recency-weighted average
        weights = np.exp(np.linspace(-1, 0, len(embeddings)))
        weights = weights / weights.sum()

        return np.average(embeddings, axis=0, weights=weights)

    def _get_recent_creators(
        self, user_profile: UserProfile, session_items: Optional[List[Item]]
    ) -> Dict[str, int]:
        """Count how many times each creator was seen recently."""
        creators = defaultdict(int)

        if session_items:
            for item in session_items[-20:]:
                if item.creator:
                    creators[item.creator] += 1

        return dict(creators)

    def _build_cross_encoder_query(
        self,
        user_profile: UserProfile,
        session_items: Optional[List[Item]],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build query for cross-encoder."""
        parts = []

        # Recent topics
        if session_items:
            recent_topics = []
            for item in session_items[-3:]:
                recent_topics.extend(item.topics)
            if recent_topics:
                parts.append(f"Recent interests: {', '.join(recent_topics[:5])}")

        # Time budget
        if context and context.get("time_budget_minutes"):
            parts.append(f"Time budget: {context['time_budget_minutes']} minutes")

        # Format preference
        if context and context.get("preferred_format"):
            parts.append(f"Preferred format: {context['preferred_format']}")

        # Avoid
        if user_profile.muted_topics:
            parts.append(f"Avoid: {', '.join(user_profile.muted_topics[:3])}")

        return ". ".join(parts) if parts else "Find relevant content"

    def _compute_recency_decay(self, item: Item) -> float:
        """Compute recency decay score."""
        if not item.published_at:
            return 0.0

        # Get half-life for this source/type
        half_life = self.recency_half_life.get(
            item.source, self.recency_half_life.get("default", 14)
        )

        # Compute age in days
        age_days = (datetime.utcnow() - item.published_at).total_seconds() / 86400

        # Exponential decay
        decay = np.exp(-np.log(2) * age_days / half_life)

        return float(decay * 0.2)  # Scale to ~0.2 max

    def _compute_repetition_penalty(
        self,
        item: Item,
        user_profile: UserProfile,
        session_items: Optional[List[Item]],
    ) -> float:
        """Penalize if item was seen very recently."""
        if not session_items:
            return 0.0

        # Check if this exact item was in last N items
        recent_ids = [it.id for it in session_items[-10:]]
        if item.id in recent_ids:
            return self.repetition_penalty

        return 0.0

    def _apply_mmr(
        self,
        items: List[tuple],
        scores: Dict[str, Dict[str, float]],
        user_profile: UserProfile,
        top_k: int,
        lambda_param: float = 0.7,
    ) -> List[tuple]:
        """Apply Maximal Marginal Relevance for diversity.

        MMR = λ * relevance - (1-λ) * max_similarity_to_selected
        """
        if not items:
            return []

        # Sort by relevance score
        sorted_items = sorted(
            items, key=lambda x: scores[x[0].id]["total"], reverse=True
        )

        selected = []
        selected_embeddings = []

        for _ in range(min(top_k, len(sorted_items))):
            if not selected:
                # First item: just take highest relevance
                best = sorted_items[0]
                selected.append(best)
                emb = self.vector_store.get_vector(best[0].id)
                if emb is not None:
                    selected_embeddings.append(emb)
                sorted_items = sorted_items[1:]
            else:
                # Compute MMR for remaining items
                best_mmr = -float("inf")
                best_idx = 0

                for idx, (item, cand) in enumerate(sorted_items):
                    relevance = scores[item.id]["total"]

                    # Compute max similarity to selected
                    item_emb = self.vector_store.get_vector(item.id)
                    if item_emb is not None and selected_embeddings:
                        similarities = [
                            np.dot(item_emb, sel_emb)
                            / (np.linalg.norm(item_emb) * np.linalg.norm(sel_emb))
                            for sel_emb in selected_embeddings
                        ]
                        max_sim = max(similarities)
                    else:
                        max_sim = 0.0

                    # MMR score
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = idx

                # Add best MMR item
                best = sorted_items[best_idx]
                selected.append(best)
                emb = self.vector_store.get_vector(best[0].id)
                if emb is not None:
                    selected_embeddings.append(emb)
                sorted_items.pop(best_idx)

        # Add scores to results
        results = []
        for item, cand in selected:
            score_data = scores[item.id]
            results.append((item, cand, score_data["total"], score_data["components"]))

        return results
