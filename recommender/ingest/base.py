"""Base ingester interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..database import Item


class BaseIngester(ABC):
    """Abstract base class for content ingesters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize ingester with configuration."""
        self.config = config

    @abstractmethod
    async def fetch_items(self) -> List[Item]:
        """Fetch new items from the source.

        Returns:
            List of Item objects
        """
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Return the source name (youtube, rss, etc.)."""
        pass
