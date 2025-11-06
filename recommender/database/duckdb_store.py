"""DuckDB implementation for relational data storage."""

import duckdb
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .schema import (
    Item,
    Event,
    UserProfile,
    CreatorPrior,
    SourcePrior,
    ExplorationArm,
    MediaType,
    EventType,
)


class DuckDBStore:
    """DuckDB store for items, events, and counters.

    Uses DuckDB for fast analytics on events and metadata.
    No training - just simple SQL aggregations and counters.
    """

    def __init__(self, db_path: str):
        """Initialize DuckDB store."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = duckdb.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create all necessary tables."""
        # Items table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id VARCHAR PRIMARY KEY,
                source VARCHAR NOT NULL,
                url VARCHAR,
                title VARCHAR,
                text VARCHAR,
                summary VARCHAR,
                creator VARCHAR,
                creator_url VARCHAR,
                media_type VARCHAR,
                tags VARCHAR,  -- JSON array
                topics VARCHAR,  -- JSON array
                length_seconds INTEGER,
                length_words INTEGER,
                time_to_value VARCHAR,
                audience_level VARCHAR,
                quality_hints VARCHAR,  -- JSON object
                published_at TIMESTAMP,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                embedding_id VARCHAR,
                extras VARCHAR  -- JSON object
            )
        """)

        # Events table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id VARCHAR PRIMARY KEY,
                item_id VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                dwell_seconds DOUBLE,
                completion_fraction DOUBLE,
                session_id VARCHAR,
                context VARCHAR  -- JSON object
            )
        """)

        # User profiles table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id VARCHAR PRIMARY KEY,
                global_embedding VARCHAR,  -- JSON array
                top_topics VARCHAR,  -- JSON array
                top_creators VARCHAR,  -- JSON array
                muted_topics VARCHAR,  -- JSON array
                muted_creators VARCHAR,  -- JSON array
                preferred_length_min INTEGER,
                preferred_length_max INTEGER,
                preferred_formats VARCHAR,  -- JSON array
                total_items_viewed INTEGER,
                total_items_completed INTEGER,
                avg_session_length_minutes DOUBLE,
                updated_at TIMESTAMP
            )
        """)

        # Creator priors table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS creator_priors (
                creator VARCHAR PRIMARY KEY,
                successes INTEGER,
                failures INTEGER,
                total_views INTEGER,
                avg_completion_rate DOUBLE,
                last_updated TIMESTAMP
            )
        """)

        # Source priors table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS source_priors (
                source VARCHAR PRIMARY KEY,
                successes INTEGER,
                failures INTEGER,
                total_views INTEGER,
                avg_reward DOUBLE,
                last_updated TIMESTAMP
            )
        """)

        # Exploration arms table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS exploration_arms (
                name VARCHAR PRIMARY KEY,
                alpha DOUBLE,
                beta DOUBLE,
                total_pulls INTEGER,
                last_pulled TIMESTAMP
            )
        """)

        # Create indices
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_source ON items(source)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_items_creator ON items(creator)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_media_type ON items(media_type)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_published ON items(published_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_item ON events(item_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)"
        )

    # === Item operations ===

    def save_item(self, item: Item):
        """Save or update an item."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO items VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """,
            [
                item.id,
                item.source,
                item.url,
                item.title,
                item.text,
                item.summary,
                item.creator,
                item.creator_url,
                item.media_type.value,
                json.dumps(item.tags),
                json.dumps(item.topics),
                item.length_seconds,
                item.length_words,
                item.time_to_value,
                item.audience_level,
                json.dumps(item.quality_hints),
                item.published_at,
                item.created_at,
                item.updated_at,
                item.embedding_id,
                json.dumps(item.extras),
            ],
        )

    def get_item(self, item_id: str) -> Optional[Item]:
        """Get item by ID."""
        result = self.conn.execute(
            "SELECT * FROM items WHERE id = ?", [item_id]
        ).fetchone()

        if not result:
            return None

        return self._row_to_item(result)

    def get_items(
        self,
        source: Optional[str] = None,
        media_type: Optional[MediaType] = None,
        limit: int = 100,
    ) -> List[Item]:
        """Get items with optional filters."""
        query = "SELECT * FROM items WHERE 1=1"
        params = []

        if source:
            query += " AND source = ?"
            params.append(source)

        if media_type:
            query += " AND media_type = ?"
            params.append(media_type.value)

        query += f" LIMIT {limit}"

        results = self.conn.execute(query, params).fetchall()
        return [self._row_to_item(row) for row in results]

    def _row_to_item(self, row) -> Item:
        """Convert DB row to Item."""
        return Item(
            id=row[0],
            source=row[1],
            url=row[2],
            title=row[3],
            text=row[4],
            summary=row[5],
            creator=row[6],
            creator_url=row[7],
            media_type=MediaType(row[8]),
            tags=json.loads(row[9]) if row[9] else [],
            topics=json.loads(row[10]) if row[10] else [],
            length_seconds=row[11],
            length_words=row[12],
            time_to_value=row[13],
            audience_level=row[14],
            quality_hints=json.loads(row[15]) if row[15] else {},
            published_at=row[16],
            created_at=row[17],
            updated_at=row[18],
            embedding_id=row[19],
            extras=json.loads(row[20]) if row[20] else {},
        )

    # === Event operations ===

    def log_event(self, event: Event):
        """Log a user event."""
        self.conn.execute(
            """
            INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                event.id,
                event.item_id,
                event.event_type.value,
                event.timestamp,
                event.dwell_seconds,
                event.completion_fraction,
                event.session_id,
                json.dumps(event.context),
            ],
        )

    def get_recent_events(
        self, limit: int = 100, event_type: Optional[EventType] = None
    ) -> List[Event]:
        """Get recent events."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        results = self.conn.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in results]

    def _row_to_event(self, row) -> Event:
        """Convert DB row to Event."""
        return Event(
            id=row[0],
            item_id=row[1],
            event_type=EventType(row[2]),
            timestamp=row[3],
            dwell_seconds=row[4],
            completion_fraction=row[5],
            session_id=row[6],
            context=json.loads(row[7]) if row[7] else {},
        )

    # === User profile operations ===

    def save_user_profile(self, profile: UserProfile):
        """Save or update user profile."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO user_profiles VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """,
            [
                profile.user_id,
                json.dumps(profile.global_embedding) if profile.global_embedding else None,
                json.dumps(profile.top_topics),
                json.dumps(profile.top_creators),
                json.dumps(profile.muted_topics),
                json.dumps(profile.muted_creators),
                profile.preferred_length_min,
                profile.preferred_length_max,
                json.dumps([f.value for f in profile.preferred_formats]),
                profile.total_items_viewed,
                profile.total_items_completed,
                profile.avg_session_length_minutes,
                profile.updated_at,
            ],
        )

    def get_user_profile(self, user_id: str = "default") -> UserProfile:
        """Get user profile."""
        result = self.conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", [user_id]
        ).fetchone()

        if not result:
            # Return default profile
            profile = UserProfile(user_id=user_id)
            self.save_user_profile(profile)
            return profile

        return UserProfile(
            user_id=result[0],
            global_embedding=json.loads(result[1]) if result[1] else None,
            top_topics=json.loads(result[2]) if result[2] else [],
            top_creators=json.loads(result[3]) if result[3] else [],
            muted_topics=json.loads(result[4]) if result[4] else [],
            muted_creators=json.loads(result[5]) if result[5] else [],
            preferred_length_min=result[6],
            preferred_length_max=result[7],
            preferred_formats=[MediaType(f) for f in json.loads(result[8])]
            if result[8]
            else [],
            total_items_viewed=result[9],
            total_items_completed=result[10],
            avg_session_length_minutes=result[11],
            updated_at=result[12],
        )

    # === Prior operations ===

    def save_creator_prior(self, prior: CreatorPrior):
        """Save or update creator prior."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO creator_priors VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                prior.creator,
                prior.successes,
                prior.failures,
                prior.total_views,
                prior.avg_completion_rate,
                prior.last_updated,
            ],
        )

    def get_creator_prior(self, creator: str) -> CreatorPrior:
        """Get creator prior."""
        result = self.conn.execute(
            "SELECT * FROM creator_priors WHERE creator = ?", [creator]
        ).fetchone()

        if not result:
            return CreatorPrior(creator=creator)

        return CreatorPrior(
            creator=result[0],
            successes=result[1],
            failures=result[2],
            total_views=result[3],
            avg_completion_rate=result[4],
            last_updated=result[5],
        )

    def save_source_prior(self, prior: SourcePrior):
        """Save or update source prior."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO source_priors VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                prior.source,
                prior.successes,
                prior.failures,
                prior.total_views,
                prior.avg_reward,
                prior.last_updated,
            ],
        )

    def get_source_prior(self, source: str) -> SourcePrior:
        """Get source prior."""
        result = self.conn.execute(
            "SELECT * FROM source_priors WHERE source = ?", [source]
        ).fetchone()

        if not result:
            return SourcePrior(source=source)

        return SourcePrior(
            source=result[0],
            successes=result[1],
            failures=result[2],
            total_views=result[3],
            avg_reward=result[4],
            last_updated=result[5],
        )

    # === Exploration arm operations ===

    def save_exploration_arm(self, arm: ExplorationArm):
        """Save or update exploration arm."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO exploration_arms VALUES (?, ?, ?, ?, ?)
        """,
            [arm.name, arm.alpha, arm.beta, arm.total_pulls, arm.last_pulled],
        )

    def get_exploration_arm(self, name: str) -> ExplorationArm:
        """Get exploration arm."""
        result = self.conn.execute(
            "SELECT * FROM exploration_arms WHERE name = ?", [name]
        ).fetchone()

        if not result:
            return ExplorationArm(name=name)

        return ExplorationArm(
            name=result[0],
            alpha=result[1],
            beta=result[2],
            total_pulls=result[3],
            last_pulled=result[4],
        )

    def get_all_exploration_arms(self) -> List[ExplorationArm]:
        """Get all exploration arms."""
        results = self.conn.execute("SELECT * FROM exploration_arms").fetchall()

        return [
            ExplorationArm(
                name=row[0],
                alpha=row[1],
                beta=row[2],
                total_pulls=row[3],
                last_pulled=row[4],
            )
            for row in results
        ]

    def close(self):
        """Close database connection."""
        self.conn.close()
