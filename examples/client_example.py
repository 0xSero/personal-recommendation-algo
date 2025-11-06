#!/usr/bin/env python3
"""Example client for the personal recommender API."""

import requests
import json
from typing import Optional


class RecommenderClient:
    """Simple client for the recommender API."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        """Initialize client."""
        self.base_url = base_url
        self.session_id = None

    def get_recommendations(
        self,
        num_items: int = 20,
        time_budget_minutes: Optional[int] = None,
        preferred_format: Optional[str] = None,
    ) -> dict:
        """Get personalized recommendations."""
        payload = {
            "num_items": num_items,
            "session_id": self.session_id,
        }

        # Add context if provided
        context = {}
        if time_budget_minutes:
            context["time_budget_minutes"] = time_budget_minutes
        if preferred_format:
            context["preferred_format"] = preferred_format

        if context:
            payload["context"] = context

        response = requests.post(f"{self.base_url}/recommend", json=payload)
        response.raise_for_status()

        data = response.json()

        # Store session ID
        self.session_id = data["session_id"]

        return data

    def log_feedback(
        self,
        item_id: str,
        event_type: str,
        dwell_seconds: Optional[float] = None,
        completion_fraction: Optional[float] = None,
    ):
        """Log feedback for an item."""
        payload = {
            "item_id": item_id,
            "event_type": event_type,
            "session_id": self.session_id,
        }

        if dwell_seconds is not None:
            payload["dwell_seconds"] = dwell_seconds
        if completion_fraction is not None:
            payload["completion_fraction"] = completion_fraction

        response = requests.post(f"{self.base_url}/feedback", json=payload)
        response.raise_for_status()

        return response.json()

    def get_stats(self) -> dict:
        """Get user statistics."""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()

    def get_item(self, item_id: str) -> dict:
        """Get item details."""
        response = requests.get(f"{self.base_url}/items/{item_id}")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage."""
    client = RecommenderClient()

    print("🎯 Personal Recommender Client Example\n")

    # Get recommendations
    print("📋 Getting recommendations...")
    recs = client.get_recommendations(
        num_items=10,
        time_budget_minutes=30,
        preferred_format="article",
    )

    print(f"\n✨ Got {len(recs['items'])} recommendations:\n")

    for i, item in enumerate(recs["items"], 1):
        print(f"{i}. {item['title']}")
        print(f"   Creator: {item['creator']}")
        print(f"   Score: {item['score']:.3f} | Strategy: {item['strategy']}")
        print(f"   Topics: {', '.join(item['topics'][:3])}")
        print(f"   URL: {item['url']}\n")

    # Simulate user interaction
    if recs["items"]:
        first_item = recs["items"][0]
        print(f"\n👆 Clicking on first item: {first_item['title']}")

        # Log click
        client.log_feedback(first_item["id"], "click")

        # Simulate reading (30 seconds, 80% completion)
        client.log_feedback(
            first_item["id"],
            "completion",
            dwell_seconds=30.0,
            completion_fraction=0.8,
        )

        print("✓ Feedback logged!")

    # Get stats
    print("\n📊 User Statistics:")
    stats = client.get_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
