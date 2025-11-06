"""Thompson Sampling for exploration (simple bandit, no models)."""

from typing import List, Dict, Any, Optional
import numpy as np

from ..database import DuckDBStore, ExplorationArm, Item, UserProfile, MediaType
from datetime import datetime, timedelta


class ThompsonSamplingExplorer:
    """Thompson Sampling exploration over predefined arms.

    Each arm represents a strategy (new_creators, deep_dives, etc.).
    Arms are just Beta distributions with (alpha, beta) counters.
    NO MODEL TRAINING - just Bayesian bandit math.
    """

    def __init__(self, db_store: DuckDBStore, config: Dict[str, Any]):
        """Initialize Thompson Sampling explorer."""
        self.db_store = db_store
        self.config = config

        # Initialize arms if not present
        self._initialize_arms()

    def _initialize_arms(self):
        """Initialize exploration arms from config."""
        arm_configs = self.config.get("arms", [])
        alpha_prior = self.config.get("alpha_prior", 1.0)
        beta_prior = self.config.get("beta_prior", 1.0)

        for arm_config in arm_configs:
            arm_name = arm_config["name"]

            # Check if arm exists
            arm = self.db_store.get_exploration_arm(arm_name)

            # Initialize with priors if new
            if arm.total_pulls == 0:
                arm.alpha = alpha_prior
                arm.beta = beta_prior
                self.db_store.save_exploration_arm(arm)

    def select_arm(self) -> str:
        """Select an arm using Thompson Sampling."""
        arms = self.db_store.get_all_exploration_arms()

        if not arms:
            return "new_creators"  # Default

        # Sample from each arm's Beta distribution
        samples = []
        for arm in arms:
            sample = arm.sample()
            samples.append((arm.name, sample))

        # Select arm with highest sample
        selected_name, _ = max(samples, key=lambda x: x[1])

        # Update last_pulled
        selected_arm = self.db_store.get_exploration_arm(selected_name)
        selected_arm.last_pulled = datetime.utcnow()
        self.db_store.save_exploration_arm(selected_arm)

        return selected_name

    def get_exploration_items(
        self,
        user_profile: UserProfile,
        num_items: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get exploration items by sampling arms."""
        results = []

        # Get arm configs
        arm_configs = {
            arm["name"]: arm for arm in self.config.get("arms", [])
        }

        # Sample arms and get items
        for _ in range(num_items):
            arm_name = self.select_arm()
            arm_config = arm_configs.get(arm_name, {})

            # Get item for this arm
            item = self._get_item_for_arm(arm_name, user_profile)

            if item:
                results.append(
                    {
                        "item_id": item.id,
                        "arm": arm_name,
                        "metadata": {
                            "source": item.source,
                            "creator": item.creator,
                            "media_type": item.media_type.value,
                            "topics": item.topics,
                        },
                    }
                )

        return results

    def _get_item_for_arm(
        self, arm_name: str, user_profile: UserProfile
    ) -> Optional[Item]:
        """Get an item matching the arm's strategy."""
        # Get recent events to avoid repetition
        recent_events = self.db_store.get_recent_events(limit=100)
        recent_item_ids = {evt.item_id for evt in recent_events}

        if arm_name == "new_creators":
            # Find items from creators user hasn't seen much
            return self._get_new_creator_item(user_profile, recent_item_ids)

        elif arm_name == "deep_dives":
            # Long-form content (>30 min)
            return self._get_deep_dive_item(user_profile, recent_item_ids)

        elif arm_name == "short_sharp":
            # Short content (<10 min)
            return self._get_short_item(user_profile, recent_item_ids)

        elif arm_name == "hot_news":
            # Very recent items (last 24 hours)
            return self._get_hot_news_item(user_profile, recent_item_ids)

        else:
            # Default: random item
            all_items = self.db_store.get_items(limit=100)
            valid = [
                item for item in all_items
                if item.id not in recent_item_ids
                and item.creator not in user_profile.muted_creators
            ]
            return valid[0] if valid else None

    def _get_new_creator_item(
        self, user_profile: UserProfile, exclude_ids: set
    ) -> Optional[Item]:
        """Get item from a new creator."""
        # Get all items
        all_items = self.db_store.get_items(limit=200)

        # Filter to new creators
        new_creator_items = []
        for item in all_items:
            if item.id in exclude_ids:
                continue
            if item.creator in user_profile.muted_creators:
                continue
            if item.creator and item.creator not in user_profile.top_creators:
                new_creator_items.append(item)

        if not new_creator_items:
            return None

        # Random choice
        return np.random.choice(new_creator_items)

    def _get_deep_dive_item(
        self, user_profile: UserProfile, exclude_ids: set
    ) -> Optional[Item]:
        """Get long-form content."""
        all_items = self.db_store.get_items(limit=200)

        deep_items = []
        for item in all_items:
            if item.id in exclude_ids:
                continue
            if item.creator in user_profile.muted_creators:
                continue
            if item.length_seconds and item.length_seconds > 1800:  # >30 min
                deep_items.append(item)

        if not deep_items:
            return None

        return np.random.choice(deep_items)

    def _get_short_item(
        self, user_profile: UserProfile, exclude_ids: set
    ) -> Optional[Item]:
        """Get short content."""
        all_items = self.db_store.get_items(limit=200)

        short_items = []
        for item in all_items:
            if item.id in exclude_ids:
                continue
            if item.creator in user_profile.muted_creators:
                continue
            if item.length_seconds and item.length_seconds < 600:  # <10 min
                short_items.append(item)

        if not short_items:
            return None

        return np.random.choice(short_items)

    def _get_hot_news_item(
        self, user_profile: UserProfile, exclude_ids: set
    ) -> Optional[Item]:
        """Get very recent items."""
        cutoff = datetime.utcnow() - timedelta(days=1)

        all_items = self.db_store.get_items(limit=200)

        hot_items = []
        for item in all_items:
            if item.id in exclude_ids:
                continue
            if item.creator in user_profile.muted_creators:
                continue
            if item.published_at and item.published_at > cutoff:
                hot_items.append(item)

        if not hot_items:
            return None

        return np.random.choice(hot_items)

    def update_arm(self, arm_name: str, success: bool):
        """Update arm with observation."""
        arm = self.db_store.get_exploration_arm(arm_name)
        arm.update(success)
        self.db_store.save_exploration_arm(arm)
