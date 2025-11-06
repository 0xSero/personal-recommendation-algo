"""Content enricher using LLM (zero-shot prompting, no training)."""

from typing import Dict, Any, List
import json
import re

from ..database import Item
from ..models import BaseLLM


class ContentEnricher:
    """Enrich items with LLM-generated summaries and metadata.

    NO TRAINING - just zero-shot prompting.
    """

    def __init__(self, llm: BaseLLM, config: Dict[str, Any]):
        """Initialize content enricher."""
        self.llm = llm
        self.config = config

        # Get topic taxonomy
        self.topics = config.get("topics", [])

        # Get prompt templates
        prompts = config.get("prompts", {})
        self.summarize_prompt = prompts.get("summarize", "")
        self.relevance_prompt = prompts.get("relevance_query", "")
        self.explanation_prompt = prompts.get("explanation", "")

    async def enrich_item(self, item: Item) -> Item:
        """Enrich an item with LLM-generated metadata.

        Args:
            item: Item to enrich

        Returns:
            Enriched item with summary, topics, etc.
        """
        # Build prompt
        prompt = self._build_summarize_prompt(item)

        # Generate summary and metadata
        try:
            response = await self.llm.agenerate(
                prompt, max_tokens=300, temperature=0.3
            )

            # Parse response
            enrichment = self._parse_enrichment_response(response)

            # Update item
            item.summary = enrichment.get("summary", "")
            item.topics = enrichment.get("topics", [])
            item.time_to_value = enrichment.get("time_to_value", "")
            item.audience_level = enrichment.get("audience_level", "")

        except Exception as e:
            print(f"Error enriching item {item.id}: {e}")

        return item

    async def generate_explanation(
        self,
        item: Item,
        user_context: Dict[str, Any],
    ) -> str:
        """Generate explanation for why item is recommended.

        Args:
            item: Recommended item
            user_context: User context (recent items, top topics, etc.)

        Returns:
            Explanation string
        """
        # Build prompt
        prompt = self.explanation_prompt.format(
            recent_items=", ".join(user_context.get("recent_items", [])),
            top_topics=", ".join(user_context.get("top_topics", [])),
            top_creators=", ".join(user_context.get("top_creators", [])),
            title=item.title,
            summary=item.summary,
            creator=item.creator or "Unknown",
        )

        try:
            explanation = await self.llm.agenerate(
                prompt, max_tokens=100, temperature=0.7
            )
            return explanation.strip()
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return "Recommended based on your interests."

    def _build_summarize_prompt(self, item: Item) -> str:
        """Build summarization prompt."""
        # Get text (limit to first 2000 chars to avoid token limits)
        text = item.text[:2000] if item.text else item.title

        # Format topics list
        topics_str = ", ".join(self.topics)

        # Fill template
        prompt = self.summarize_prompt.format(
            topics=topics_str, title=item.title, text=text
        )

        return prompt

    def _parse_enrichment_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract enrichment data."""
        result = {}

        # Extract summary (bullet points)
        bullet_pattern = r"[•\-\*]\s*(.+?)(?=\n[•\-\*]|\n\n|\Z)"
        bullets = re.findall(bullet_pattern, response, re.DOTALL)
        if bullets:
            result["summary"] = " ".join([b.strip() for b in bullets[:5]])
        else:
            # Fallback: take first paragraph
            paragraphs = response.split("\n\n")
            result["summary"] = paragraphs[0][:300] if paragraphs else ""

        # Extract time-to-value
        ttv_match = re.search(
            r"time-to-value:?\s*(.+?)(?=\n|\Z)", response, re.IGNORECASE
        )
        if ttv_match:
            result["time_to_value"] = ttv_match.group(1).strip()

        # Extract audience level
        audience_match = re.search(
            r"audience:?\s*(novice|intermediate|advanced)",
            response,
            re.IGNORECASE,
        )
        if audience_match:
            result["audience_level"] = audience_match.group(1).lower()

        # Extract tags/topics
        tags_match = re.search(r"tags?:?\s*(.+?)(?=\n|\Z)", response, re.IGNORECASE)
        if tags_match:
            tags_str = tags_match.group(1)
            # Split by commas, semicolons, or newlines
            tags = re.split(r"[,;\n]", tags_str)
            result["topics"] = [t.strip() for t in tags if t.strip()][:5]
        else:
            result["topics"] = []

        return result
