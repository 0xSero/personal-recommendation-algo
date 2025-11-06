"""Contextual recommendations based on time, activity, and mood.

NO TRAINING - just smart heuristics based on time of day, day of week, etc.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ..database import MediaType


class Mode(str, Enum):
    """Context modes."""

    MORNING_BRIEFING = "morning_briefing"
    COMMUTE = "commute"
    LUNCH_BREAK = "lunch_break"
    DEEP_WORK = "deep_work"
    EVENING_UNWIND = "evening_unwind"
    WEEKEND_EXPLORE = "weekend_explore"
    BEDTIME = "bedtime"


class ContextDetector:
    """Detect user's current context.

    Uses simple rules - NO machine learning required!
    """

    @staticmethod
    def detect_context(
        now: Optional[datetime] = None,
        device: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect current context and return recommendation parameters.

        Args:
            now: Current datetime (defaults to now)
            device: Device type (mobile, tablet, desktop)
            location: Location hint (home, work, commute)

        Returns:
            Context dict with mode, time_budget, format preferences, etc.
        """
        if now is None:
            now = datetime.now()

        hour = now.hour
        day = now.weekday()  # 0 = Monday, 6 = Sunday
        is_weekend = day >= 5

        # Morning (6 AM - 9 AM)
        if 6 <= hour < 9 and not is_weekend:
            return {
                "mode": Mode.MORNING_BRIEFING,
                "time_budget_minutes": 15,
                "preferred_formats": [MediaType.ARTICLE.value],
                "max_length_seconds": 5 * 60,
                "topics_boost": ["news", "technology", "business"],
                "recency_weight": 1.5,  # Prefer fresh content
                "novelty_weight": 0.5,  # Less exploration, more curation
                "explanation": "Quick morning reads to start your day",
            }

        # Commute (7-9 AM, 5-7 PM) - if mobile
        elif (7 <= hour < 9 or 17 <= hour < 19) and device == "mobile":
            return {
                "mode": Mode.COMMUTE,
                "time_budget_minutes": 30,
                "preferred_formats": [MediaType.PODCAST.value, MediaType.VIDEO.value],
                "min_length_seconds": 10 * 60,
                "max_length_seconds": 30 * 60,
                "topics_boost": ["educational", "entertaining"],
                "explanation": "Audio/video content for your commute",
            }

        # Lunch break (12 PM - 1 PM)
        elif 12 <= hour < 13:
            return {
                "mode": Mode.LUNCH_BREAK,
                "time_budget_minutes": 20,
                "preferred_formats": [MediaType.ARTICLE.value, MediaType.VIDEO.value],
                "max_length_seconds": 10 * 60,
                "novelty_weight": 1.2,  # More exploration
                "topics_avoid": ["work-related"],  # Take a break from work
                "explanation": "Light, interesting reads for your lunch break",
            }

        # Deep work hours (10 AM - 4 PM) - if at work
        elif 10 <= hour < 16 and location == "work":
            return {
                "mode": Mode.DEEP_WORK,
                "time_budget_minutes": 60,
                "preferred_formats": [MediaType.ARTICLE.value],
                "audience_level": "advanced",
                "depth_weight": 1.5,  # Prefer in-depth content
                "topics_boost": ["technical", "research", "professional"],
                "explanation": "In-depth professional content for focused learning",
            }

        # Evening unwind (6 PM - 10 PM)
        elif 18 <= hour < 22:
            return {
                "mode": Mode.EVENING_UNWIND,
                "time_budget_minutes": 90,
                "preferred_formats": [
                    MediaType.VIDEO.value,
                    MediaType.PODCAST.value,
                    MediaType.ARTICLE.value,
                ],
                "min_length_seconds": 15 * 60,
                "topics_boost": ["educational", "documentary", "narrative"],
                "diversity_weight": 1.3,  # More diverse recommendations
                "explanation": "Long-form content for evening deep dives",
            }

        # Bedtime (10 PM - midnight)
        elif 22 <= hour < 24:
            return {
                "mode": Mode.BEDTIME,
                "time_budget_minutes": 30,
                "preferred_formats": [MediaType.ARTICLE.value],
                "max_length_seconds": 15 * 60,
                "topics_boost": ["calming", "philosophical", "narrative"],
                "topics_avoid": ["news", "controversial"],
                "explanation": "Relaxing reads to wind down before sleep",
            }

        # Weekend
        elif is_weekend:
            return {
                "mode": Mode.WEEKEND_EXPLORE,
                "time_budget_minutes": 180,
                "preferred_formats": [
                    MediaType.VIDEO.value,
                    MediaType.ARTICLE.value,
                    MediaType.PODCAST.value,
                ],
                "novelty_weight": 1.5,  # Maximum exploration
                "topics_boost": ["hobby", "entertainment", "creative"],
                "topics_avoid": ["work-related"],
                "exploration_arms": [
                    "new_creators",
                    "new_topics",
                ],  # More exploration
                "explanation": "Weekend discoveries - explore something new",
            }

        # Default (work hours)
        else:
            return {
                "mode": "default",
                "time_budget_minutes": 30,
                "preferred_formats": [MediaType.ARTICLE.value],
                "explanation": "Recommended for you",
            }


class ContextualScorer:
    """Adjust recommendation scores based on context.

    Modifies the base reranking scores with context-aware bonuses.
    """

    @staticmethod
    def apply_context_boost(
        base_score: float,
        item: "Item",
        context: Dict[str, Any],
    ) -> float:
        """Apply context-based score adjustments.

        Args:
            base_score: Base recommendation score
            item: The item being scored
            context: Context dict from ContextDetector

        Returns:
            Adjusted score
        """
        score = base_score

        # Format preference boost
        preferred_formats = context.get("preferred_formats", [])
        if item.media_type.value in preferred_formats:
            score *= 1.2

        # Length filtering
        max_length = context.get("max_length_seconds")
        min_length = context.get("min_length_seconds")

        if max_length and item.length_seconds and item.length_seconds > max_length:
            score *= 0.5  # Penalty for too long

        if min_length and item.length_seconds and item.length_seconds < min_length:
            score *= 0.7  # Penalty for too short

        # Topic boost
        topics_boost = context.get("topics_boost", [])
        if any(topic.lower() in [t.lower() for t in item.topics] for topic in topics_boost):
            score *= 1.3

        # Topic avoid
        topics_avoid = context.get("topics_avoid", [])
        if any(topic.lower() in [t.lower() for t in item.topics] for topic in topics_avoid):
            score *= 0.3  # Strong penalty

        # Audience level match
        target_level = context.get("audience_level")
        if target_level and item.audience_level == target_level:
            score *= 1.15

        # Recency boost
        recency_weight = context.get("recency_weight", 1.0)
        if item.published_at:
            age_hours = (datetime.utcnow() - item.published_at).total_seconds() / 3600
            if age_hours < 24:  # Less than 24 hours old
                score *= recency_weight

        return score


# Example usage in API:
"""
@app.post("/recommend/contextual")
async def get_contextual_recommendations(
    num_items: int = 20,
    device: Optional[str] = None,
    location: Optional[str] = None
):
    # Detect context
    context = ContextDetector.detect_context(
        device=device,
        location=location
    )

    # Get base recommendations
    candidates = await candidate_generator.generate_candidates(
        user_profile,
        num_candidates=500,
        context=context
    )

    # Rerank with context
    ranked = await reranker.rerank(
        candidates,
        user_profile,
        context=context
    )

    # Apply additional context scoring
    for item_dict in ranked:
        item = db.get_item(item_dict["item_id"])
        item_dict["score"] = ContextualScorer.apply_context_boost(
            item_dict["score"],
            item,
            context
        )

    # Re-sort
    ranked.sort(key=lambda x: x["score"], reverse=True)

    return {
        "mode": context["mode"],
        "explanation": context["explanation"],
        "time_budget": context.get("time_budget_minutes"),
        "items": ranked[:num_items]
    }
"""


# Advanced: Activity-based context
class ActivityContext:
    """Context based on user's current activity."""

    CONTEXTS = {
        "coding": {
            "topics_boost": ["programming", "software", "technical"],
            "preferred_formats": ["article", "documentation"],
            "max_length_seconds": 10 * 60,
            "explanation": "Quick technical references while coding",
        },
        "exercising": {
            "preferred_formats": ["podcast", "audio"],
            "min_length_seconds": 20 * 60,
            "topics_boost": ["motivational", "educational"],
            "explanation": "Long-form audio for your workout",
        },
        "cooking": {
            "preferred_formats": ["video"],
            "max_length_seconds": 15 * 60,
            "topics_boost": ["food", "recipe", "cooking"],
            "explanation": "Cooking videos and recipes",
        },
        "researching": {
            "preferred_formats": ["article", "paper"],
            "audience_level": "advanced",
            "depth_weight": 2.0,
            "topics_boost": ["research", "academic", "analysis"],
            "explanation": "Deep research materials",
        },
        "relaxing": {
            "preferred_formats": ["video", "article"],
            "topics_boost": ["entertainment", "light", "fun"],
            "topics_avoid": ["news", "controversial", "technical"],
            "explanation": "Relaxing content to unwind",
        },
    }

    @classmethod
    def get_activity_context(cls, activity: str) -> Dict[str, Any]:
        """Get context for a specific activity."""
        return cls.CONTEXTS.get(activity, {})


# Mood-based context
class MoodContext:
    """Context based on user's mood."""

    MOODS = {
        "curious": {
            "novelty_weight": 2.0,
            "exploration_arms": ["new_topics", "new_creators"],
            "diversity_weight": 1.5,
            "explanation": "Exploring new and diverse topics",
        },
        "focused": {
            "audience_level": "advanced",
            "depth_weight": 1.8,
            "min_length_seconds": 30 * 60,
            "explanation": "Deep, focused content for serious learning",
        },
        "casual": {
            "max_length_seconds": 10 * 60,
            "topics_boost": ["light", "entertaining"],
            "explanation": "Quick, easy reads",
        },
        "inspired": {
            "topics_boost": ["creative", "motivational", "innovative"],
            "preferred_formats": ["video", "article"],
            "explanation": "Inspiring and creative content",
        },
    }

    @classmethod
    def get_mood_context(cls, mood: str) -> Dict[str, Any]:
        """Get context for a specific mood."""
        return cls.MOODS.get(mood, {})


# Combine multiple contexts
class ContextCombiner:
    """Combine multiple context sources."""

    @staticmethod
    def combine(
        time_context: Dict[str, Any],
        activity_context: Optional[Dict[str, Any]] = None,
        mood_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Combine multiple contexts intelligently."""
        combined = time_context.copy()

        # Merge activity context
        if activity_context:
            # Activity overrides some time-based preferences
            combined.update(activity_context)

        # Merge mood context
        if mood_context:
            # Mood adjusts weights
            for key in ["novelty_weight", "depth_weight", "diversity_weight"]:
                if key in mood_context:
                    combined[key] = mood_context[key]

        return combined


"""
Example API usage:

@app.post("/recommend/smart")
async def smart_recommendations(
    num_items: int = 20,
    activity: Optional[str] = None,  # "coding", "exercising", etc.
    mood: Optional[str] = None,      # "curious", "focused", etc.
    device: Optional[str] = None,
    location: Optional[str] = None
):
    # Get all contexts
    time_ctx = ContextDetector.detect_context(device=device, location=location)
    activity_ctx = ActivityContext.get_activity_context(activity) if activity else None
    mood_ctx = MoodContext.get_mood_context(mood) if mood else None

    # Combine
    context = ContextCombiner.combine(time_ctx, activity_ctx, mood_ctx)

    # Get recommendations with combined context
    recommendations = await get_recommendations_with_context(context, num_items)

    return {
        "context": {
            "time": time_ctx.get("mode"),
            "activity": activity,
            "mood": mood,
        },
        "explanation": context.get("explanation"),
        "items": recommendations
    }
"""
