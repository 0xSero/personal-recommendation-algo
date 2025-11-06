"""Data models and schema definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class MediaType(str, Enum):
    """Media type enum."""

    VIDEO = "video"
    ARTICLE = "article"
    PODCAST = "podcast"
    NOTE = "note"
    TWEET = "tweet"
    OTHER = "other"


class EventType(str, Enum):
    """User event types."""

    CLICK = "click"
    DWELL = "dwell"
    COMPLETION = "completion"
    LIKE = "like"
    DISLIKE = "dislike"
    SAVE = "save"
    HIDE = "hide"
    SHARE = "share"


@dataclass
class Item:
    """A content item (video, article, podcast, etc.)."""

    # Core fields
    id: str
    source: str  # youtube, rss, browser, local, podcast
    url: Optional[str] = None
    title: str = ""
    text: str = ""  # Full text / transcript
    summary: str = ""  # LLM-generated summary
    creator: Optional[str] = None
    creator_url: Optional[str] = None

    # Metadata
    media_type: MediaType = MediaType.OTHER
    tags: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    length_seconds: Optional[int] = None
    length_words: Optional[int] = None

    # Enrichment (LLM-generated)
    time_to_value: Optional[str] = None
    audience_level: Optional[str] = None  # novice, intermediate, advanced
    quality_hints: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    published_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Vector embedding (stored separately in vector DB)
    embedding_id: Optional[str] = None

    # Extra fields (source-specific metadata)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "summary": self.summary,
            "creator": self.creator,
            "creator_url": self.creator_url,
            "media_type": self.media_type.value,
            "tags": self.tags,
            "topics": self.topics,
            "length_seconds": self.length_seconds,
            "length_words": self.length_words,
            "time_to_value": self.time_to_value,
            "audience_level": self.audience_level,
            "quality_hints": self.quality_hints,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "embedding_id": self.embedding_id,
            "extras": self.extras,
        }


@dataclass
class Event:
    """User interaction event."""

    id: str
    item_id: str
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Event-specific data
    dwell_seconds: Optional[float] = None
    completion_fraction: Optional[float] = None
    session_id: Optional[str] = None

    # Context at time of event
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "item_id": self.item_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "dwell_seconds": self.dwell_seconds,
            "completion_fraction": self.completion_fraction,
            "session_id": self.session_id,
            "context": self.context,
        }


@dataclass
class UserProfile:
    """User profile with preferences and history."""

    user_id: str = "default"

    # Taste representation (updated incrementally)
    global_embedding: Optional[List[float]] = None
    top_topics: List[str] = field(default_factory=list)
    top_creators: List[str] = field(default_factory=list)
    muted_topics: List[str] = field(default_factory=list)
    muted_creators: List[str] = field(default_factory=list)

    # Preferences
    preferred_length_min: Optional[int] = None
    preferred_length_max: Optional[int] = None
    preferred_formats: List[MediaType] = field(default_factory=list)

    # Stats
    total_items_viewed: int = 0
    total_items_completed: int = 0
    avg_session_length_minutes: float = 0.0

    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "global_embedding": self.global_embedding,
            "top_topics": self.top_topics,
            "top_creators": self.top_creators,
            "muted_topics": self.muted_topics,
            "muted_creators": self.muted_creators,
            "preferred_length_min": self.preferred_length_min,
            "preferred_length_max": self.preferred_length_max,
            "preferred_formats": [f.value for f in self.preferred_formats],
            "total_items_viewed": self.total_items_viewed,
            "total_items_completed": self.total_items_completed,
            "avg_session_length_minutes": self.avg_session_length_minutes,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class CreatorPrior:
    """Prior belief about creator quality (simple counters)."""

    creator: str
    successes: int = 0  # completed items
    failures: int = 0  # bounced/hidden items
    total_views: int = 0
    avg_completion_rate: float = 0.5
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update(self, success: bool, decay: float = 0.95):
        """Update prior with new observation."""
        if success:
            self.successes += 1
        else:
            self.failures += 1

        self.total_views += 1

        # Exponential moving average
        new_rate = self.successes / max(self.total_views, 1)
        self.avg_completion_rate = (
            decay * self.avg_completion_rate + (1 - decay) * new_rate
        )
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "creator": self.creator,
            "successes": self.successes,
            "failures": self.failures,
            "total_views": self.total_views,
            "avg_completion_rate": self.avg_completion_rate,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class SourcePrior:
    """Prior belief about source quality (simple counters)."""

    source: str
    successes: int = 0
    failures: int = 0
    total_views: int = 0
    avg_reward: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update(self, reward: float, decay: float = 0.95):
        """Update prior with new reward."""
        if reward > 0.5:
            self.successes += 1
        else:
            self.failures += 1

        self.total_views += 1

        # Exponential moving average
        self.avg_reward = decay * self.avg_reward + (1 - decay) * reward
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "successes": self.successes,
            "failures": self.failures,
            "total_views": self.total_views,
            "avg_reward": self.avg_reward,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class ExplorationArm:
    """Thompson Sampling arm for exploration."""

    name: str
    alpha: float = 1.0  # successes + prior
    beta: float = 1.0  # failures + prior
    total_pulls: int = 0
    last_pulled: Optional[datetime] = None

    def sample(self) -> float:
        """Sample from Beta distribution."""
        import numpy as np

        return np.random.beta(self.alpha, self.beta)

    def update(self, success: bool):
        """Update arm with observation."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1

        self.total_pulls += 1
        self.last_pulled = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "total_pulls": self.total_pulls,
            "last_pulled": self.last_pulled.isoformat() if self.last_pulled else None,
        }
