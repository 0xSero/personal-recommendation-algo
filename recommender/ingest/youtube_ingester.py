"""YouTube ingester using yt-dlp."""

from typing import List, Dict, Any
import asyncio
from datetime import datetime
import hashlib

from .base import BaseIngester
from ..database import Item, MediaType


class YouTubeIngester(BaseIngester):
    """Ingest YouTube videos using yt-dlp."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize YouTube ingester."""
        super().__init__(config)
        self.max_videos = config.get("max_videos_per_poll", 100)
        self.download_captions = config.get("download_captions", True)

    async def fetch_items(self) -> List[Item]:
        """Fetch YouTube videos.

        This is a placeholder - in production you'd:
        1. Use yt-dlp to fetch from subscriptions/channels
        2. Extract metadata, captions, etc.
        3. Store videos locally if needed
        """
        # Example: would use yt-dlp here
        # For now, return empty list
        # In production:
        # - Read channel list from config
        # - Use yt-dlp to fetch recent videos
        # - Extract metadata and captions
        return []

    def get_source_name(self) -> str:
        """Return source name."""
        return "youtube"

    async def _fetch_channel_videos(
        self, channel_url: str, limit: int = 50
    ) -> List[Item]:
        """Fetch videos from a YouTube channel.

        Args:
            channel_url: URL of the channel
            limit: Maximum number of videos to fetch

        Returns:
            List of Item objects
        """
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "playlistend": limit,
        }

        items = []

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Extract channel/playlist info
                info = ydl.extract_info(channel_url, download=False)

                if not info:
                    return items

                # Process entries
                entries = info.get("entries", [])

                for entry in entries[:limit]:
                    video_id = entry.get("id")
                    if not video_id:
                        continue

                    # Create item
                    item = Item(
                        id=f"youtube_{video_id}",
                        source="youtube",
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        title=entry.get("title", ""),
                        creator=entry.get("uploader", ""),
                        creator_url=entry.get("uploader_url", ""),
                        media_type=MediaType.VIDEO,
                        length_seconds=entry.get("duration"),
                        published_at=self._parse_upload_date(entry.get("upload_date")),
                        extras={
                            "view_count": entry.get("view_count"),
                            "like_count": entry.get("like_count"),
                            "channel_id": entry.get("channel_id"),
                        },
                    )

                    items.append(item)

            except Exception as e:
                print(f"Error fetching YouTube channel: {e}")

        return items

    def _parse_upload_date(self, upload_date: str) -> datetime:
        """Parse YouTube upload date (YYYYMMDD format)."""
        if not upload_date:
            return datetime.utcnow()

        try:
            return datetime.strptime(upload_date, "%Y%m%d")
        except:
            return datetime.utcnow()
