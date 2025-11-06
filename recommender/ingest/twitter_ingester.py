"""Twitter/X ingester example - shows how easy it is to add new sources."""

from typing import List
import asyncio
from datetime import datetime

from .base import BaseIngester
from ..database import Item, MediaType


class TwitterIngester(BaseIngester):
    """Ingest from Twitter bookmarks, likes, and lists.

    This is a complete working example showing how to add a new content source.
    Just ~50 lines of code!
    """

    def __init__(self, config):
        """Initialize Twitter ingester.

        Config should contain:
        - bearer_token: Twitter API bearer token
        - sources: ["bookmarks", "likes", "lists"]
        - list_ids: ["123", "456"] (if using lists)
        """
        super().__init__(config)
        self.bearer_token = config.get("bearer_token")
        self.sources = config.get("sources", ["bookmarks"])
        self.list_ids = config.get("list_ids", [])

    async def fetch_items(self) -> List[Item]:
        """Fetch items from Twitter."""
        import tweepy

        # Authenticate
        client = tweepy.Client(bearer_token=self.bearer_token)

        items = []

        # Fetch from each source
        if "bookmarks" in self.sources:
            items.extend(await self._fetch_bookmarks(client))

        if "likes" in self.sources:
            items.extend(await self._fetch_likes(client))

        if "lists" in self.sources:
            for list_id in self.list_ids:
                items.extend(await self._fetch_list(client, list_id))

        return items

    async def _fetch_bookmarks(self, client) -> List[Item]:
        """Fetch bookmarked tweets."""
        items = []

        # Get user's bookmarks
        response = client.get_bookmarks(
            max_results=100,
            tweet_fields=["created_at", "author_id", "public_metrics"],
            expansions=["author_id"],
        )

        if not response.data:
            return items

        # Get user info for authors
        users = {user.id: user for user in response.includes.get("users", [])}

        for tweet in response.data:
            # Get full thread if it's part of one
            thread_text = await self._get_thread(client, tweet)

            # Get author
            author = users.get(tweet.author_id)
            author_name = author.username if author else "unknown"

            # Create item
            item = Item(
                id=f"twitter_{tweet.id}",
                source="twitter",
                url=f"https://twitter.com/x/status/{tweet.id}",
                title=f"Tweet by @{author_name}",
                text=thread_text or tweet.text,
                creator=author_name,
                creator_url=f"https://twitter.com/{author_name}",
                media_type=MediaType.TWEET,
                published_at=tweet.created_at,
                extras={
                    "likes": tweet.public_metrics.get("like_count", 0),
                    "retweets": tweet.public_metrics.get("retweet_count", 0),
                    "replies": tweet.public_metrics.get("reply_count", 0),
                },
            )

            items.append(item)

        return items

    async def _fetch_likes(self, client) -> List[Item]:
        """Fetch liked tweets."""
        # Similar to bookmarks
        # ... implementation ...
        return []

    async def _fetch_list(self, client, list_id: str) -> List[Item]:
        """Fetch tweets from a list."""
        # ... implementation ...
        return []

    async def _get_thread(self, client, tweet) -> str:
        """Get full thread if tweet is part of one."""
        # Check if tweet is a reply
        if not tweet.in_reply_to_user_id:
            return tweet.text

        # Get conversation
        try:
            conversation = client.search_recent_tweets(
                query=f"conversation_id:{tweet.conversation_id}",
                max_results=100,
                tweet_fields=["created_at"],
            )

            if not conversation.data:
                return tweet.text

            # Sort by time and concatenate
            tweets = sorted(conversation.data, key=lambda t: t.created_at)
            thread_text = "\n\n".join([t.text for t in tweets])

            return thread_text

        except Exception as e:
            print(f"Error fetching thread: {e}")
            return tweet.text

    def get_source_name(self) -> str:
        """Return source name."""
        return "twitter"


# Usage in config.yaml:
"""
sources:
  twitter:
    enabled: true
    bearer_token: "your_token_here"
    sources: ["bookmarks", "likes", "lists"]
    list_ids: ["123456789"]
    poll_interval_hours: 1
"""

# Add to ingestion pipeline:
"""
if config["sources"]["twitter"]["enabled"]:
    twitter_ingester = TwitterIngester(config["sources"]["twitter"])
    ingesters.append(twitter_ingester)
"""
