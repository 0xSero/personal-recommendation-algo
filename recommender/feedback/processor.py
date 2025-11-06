"""Feedback processor - updates counters and priors (NO TRAINING)."""

from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from ..database import (
    DuckDBStore,
    Event,
    EventType,
    Item,
    UserProfile,
    CreatorPrior,
    SourcePrior,
)


class FeedbackProcessor:
    """Process user feedback and update simple counters.

    NO TRAINING - just arithmetic:
    - Compute rewards from events
    - Update exponential moving averages
    - Update creator/source priors
    """

    def __init__(self, db_store: DuckDBStore, config: Dict[str, Any]):
        """Initialize feedback processor."""
        self.db_store = db_store
        self.config = config

        # Reward formula weights
        reward_formula = self.config.get("reward_formula", {})
        self.completion_weight = reward_formula.get("completion_weight", 0.6)
        self.dwell_weight = reward_formula.get("dwell_weight", 0.2)
        self.like_weight = reward_formula.get("like_weight", 0.2)
        self.bounce_penalty = reward_formula.get("bounce_penalty", -0.3)
        self.dwell_cap_seconds = reward_formula.get("dwell_cap_seconds", 120)

        # Prior update params
        self.prior_decay = self.config.get("prior_decay", 0.95)
        self.min_samples_for_prior = self.config.get("min_samples_for_prior", 3)

    def log_event(
        self,
        item_id: str,
        event_type: EventType,
        dwell_seconds: Optional[float] = None,
        completion_fraction: Optional[float] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Log a user event."""
        event = Event(
            id=str(uuid.uuid4()),
            item_id=item_id,
            event_type=event_type,
            dwell_seconds=dwell_seconds,
            completion_fraction=completion_fraction,
            session_id=session_id,
            context=context or {},
        )

        self.db_store.log_event(event)

        # Process feedback if it's a completion/bounce event
        if event_type in [EventType.COMPLETION, EventType.HIDE, EventType.DISLIKE]:
            self.process_feedback(event)

        return event

    def process_feedback(self, event: Event):
        """Process feedback event and update priors."""
        item = self.db_store.get_item(event.item_id)
        if not item:
            return

        # Compute reward
        reward = self._compute_reward(event, item)

        # Update creator prior
        if item.creator:
            self._update_creator_prior(item.creator, reward, event)

        # Update source prior
        self._update_source_prior(item.source, reward)

        # Update user profile
        self._update_user_profile(event, item, reward)

    def _compute_reward(self, event: Event, item: Item) -> float:
        """Compute reward from event.

        Reward = 0.6*completion + 0.2*min(dwell/120, 1) + 0.2*like - 0.3*bounce
        """
        reward = 0.0

        # Completion component
        if event.completion_fraction is not None:
            reward += self.completion_weight * event.completion_fraction
        elif event.event_type == EventType.COMPLETION:
            reward += self.completion_weight * 1.0

        # Dwell component
        if event.dwell_seconds is not None:
            dwell_score = min(event.dwell_seconds / self.dwell_cap_seconds, 1.0)
            reward += self.dwell_weight * dwell_score

        # Like component
        if event.event_type == EventType.LIKE:
            reward += self.like_weight * 1.0

        # Bounce/hide penalty
        if event.event_type in [EventType.HIDE, EventType.DISLIKE]:
            reward += self.bounce_penalty

        # Quick bounce penalty
        if event.dwell_seconds is not None and event.dwell_seconds < 10:
            reward += self.bounce_penalty * 0.5

        # Clamp to [0, 1]
        reward = max(0.0, min(1.0, reward))

        return reward

    def _update_creator_prior(
        self, creator: str, reward: float, event: Event
    ):
        """Update creator prior with exponential moving average."""
        prior = self.db_store.get_creator_prior(creator)

        # Determine success/failure
        success = reward > 0.5 or event.event_type == EventType.COMPLETION

        # Update
        prior.update(success, decay=self.prior_decay)

        # Save
        self.db_store.save_creator_prior(prior)

    def _update_source_prior(self, source: str, reward: float):
        """Update source prior with exponential moving average."""
        prior = self.db_store.get_source_prior(source)

        # Update
        prior.update(reward, decay=self.prior_decay)

        # Save
        self.db_store.save_source_prior(prior)

    def _update_user_profile(self, event: Event, item: Item, reward: float):
        """Update user profile based on event."""
        profile = self.db_store.get_user_profile()

        # Increment view counter
        profile.total_items_viewed += 1

        # Increment completion counter if completed
        if event.event_type == EventType.COMPLETION or reward > 0.7:
            profile.total_items_completed += 1

        # Update top topics (simple frequency count)
        # In production, you'd use a more sophisticated approach
        if item.topics and reward > 0.5:
            topic_counts = {}
            for topic in profile.top_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 0.5

            for topic in item.topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            # Sort and take top 10
            sorted_topics = sorted(
                topic_counts.items(), key=lambda x: x[1], reverse=True
            )
            profile.top_topics = [topic for topic, _ in sorted_topics[:10]]

        # Update top creators
        if item.creator and reward > 0.5:
            if item.creator not in profile.top_creators:
                profile.top_creators.insert(0, item.creator)
                profile.top_creators = profile.top_creators[:10]

        # Save
        self.db_store.save_user_profile(profile)

    def compute_session_reward(self, session_id: str) -> float:
        """Compute aggregate reward for a session (for exploration arms)."""
        events = self.db_store.get_recent_events(limit=1000)

        session_events = [e for e in events if e.session_id == session_id]

        if not session_events:
            return 0.0

        total_reward = 0.0
        for event in session_events:
            item = self.db_store.get_item(event.item_id)
            if item:
                reward = self._compute_reward(event, item)
                total_reward += reward

        # Average reward
        return total_reward / len(session_events) if session_events else 0.0
