"""RSS feed ingester."""

from typing import List, Dict, Any
import asyncio
import feedparser
from datetime import datetime
import hashlib

from .base import BaseIngester
from ..database import Item, MediaType


class RSSIngester(BaseIngester):
    """Ingest content from RSS feeds."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize RSS ingester."""
        super().__init__(config)
        self.feeds_file = config.get("feeds_file", "./config/rss_feeds.txt")

    async def fetch_items(self) -> List[Item]:
        """Fetch items from RSS feeds."""
        # Read feed URLs
        feed_urls = self._read_feeds_file()

        # Fetch all feeds in parallel
        tasks = [self._fetch_feed(url) for url in feed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect items
        items = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)

        return items

    def get_source_name(self) -> str:
        """Return source name."""
        return "rss"

    def _read_feeds_file(self) -> List[str]:
        """Read feed URLs from file."""
        try:
            with open(self.feeds_file, "r") as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            return urls
        except FileNotFoundError:
            return []

    async def _fetch_feed(self, feed_url: str) -> List[Item]:
        """Fetch a single RSS feed."""
        items = []

        try:
            # Parse feed (runs in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, feed_url)

            # Extract feed title for creator
            feed_title = feed.feed.get("title", feed_url)

            # Process entries
            for entry in feed.entries:
                # Generate ID from URL or guid
                item_url = entry.get("link", entry.get("id", ""))
                item_id = f"rss_{hashlib.md5(item_url.encode()).hexdigest()}"

                # Get text content
                text = entry.get("summary", entry.get("description", ""))

                # Get published date
                published_at = self._parse_date(entry)

                # Create item
                item = Item(
                    id=item_id,
                    source="rss",
                    url=item_url,
                    title=entry.get("title", ""),
                    text=text,
                    creator=entry.get("author", feed_title),
                    media_type=MediaType.ARTICLE,
                    published_at=published_at,
                    extras={
                        "feed_url": feed_url,
                        "feed_title": feed_title,
                    },
                )

                items.append(item)

        except Exception as e:
            print(f"Error fetching RSS feed {feed_url}: {e}")

        return items

    def _parse_date(self, entry) -> datetime:
        """Parse entry date."""
        # Try published_parsed
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:6])
            except:
                pass

        # Try updated_parsed
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            try:
                return datetime(*entry.updated_parsed[:6])
            except:
                pass

        # Default to now
        return datetime.utcnow()
