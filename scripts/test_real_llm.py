#!/usr/bin/env python3
"""Test with REAL OpenRouter API (minimax/minimax-m2:free)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from recommender.models import ModelFactory, BaseLLM, ModelConfig


# Create OpenRouter LLM
class OpenRouterLLM(BaseLLM):
    """OpenRouter API implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from openai import AsyncOpenAI, OpenAI

        api_key = config.extra_params.get("api_key")

        # OpenRouter requires additional headers
        default_headers = {
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Personal Recommender"
        }

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers=default_headers
        )

        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers=default_headers
        )

    def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
        """Sync generate."""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or 500,
            temperature=temperature or 0.7,
        )
        return response.choices[0].message.content

    async def agenerate(self, prompt, max_tokens=None, temperature=None, **kwargs):
        """Async generate."""
        response = await self.async_client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or 500,
            temperature=temperature or 0.7,
        )
        return response.choices[0].message.content


async def test_real_llm():
    """Test with real OpenRouter API."""

    print("🧪 Testing with REAL LLM (OpenRouter minimax-m2)\n")
    print("=" * 60)

    # Register provider
    ModelFactory.register_llm_provider("openrouter", OpenRouterLLM)

    # Create LLM
    config = {
        "provider": "openrouter",
        "model_name": "minimax/minimax-m2:free",
        "api_key": "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"
    }

    llm = ModelFactory.create_llm(config)
    print(f"✓ Created LLM: {llm}\n")

    # Test 1: Simple generation
    print("Test 1: Simple Generation")
    print("-" * 60)
    prompt = "What is machine learning? Answer in 2 sentences."

    print(f"Prompt: {prompt}")
    print("\nGenerating...")

    response = await llm.agenerate(prompt, max_tokens=100)

    print(f"\n✓ Response:\n{response}\n")

    # Test 2: Content summarization (like we'd use it)
    print("\nTest 2: Content Summarization")
    print("-" * 60)

    article = """
    Title: The Future of Large Language Models

    Large language models (LLMs) have revolutionized natural language processing.
    These models, trained on vast amounts of text data, can generate human-like text,
    answer questions, and perform various language tasks. Recent developments include
    models with hundreds of billions of parameters, capable of few-shot learning and
    reasoning. However, challenges remain around computational costs, bias, and safety.
    Researchers are exploring ways to make these models more efficient and controllable.
    """

    summary_prompt = f"""
    Summarize this article in 3 bullet points.
    Then add: Time-to-value (1 sentence), Audience level (novice/intermediate/advanced),
    and 3 tags from: [AI, Machine Learning, Technology, Research, Software].

    Article:
    {article}
    """

    print("Prompt: Summarize article...")
    print("\nGenerating summary...")

    summary = await llm.agenerate(summary_prompt, max_tokens=300)

    print(f"\n✓ Summary:\n{summary}\n")

    # Test 3: Recommendation explanation
    print("\nTest 3: Recommendation Explanation")
    print("-" * 60)

    explain_prompt = """
    Why am I recommending this article to the user?

    User profile:
    - Recently read: "Introduction to Neural Networks", "Deep Learning Basics"
    - Top topics: Machine Learning, AI, Python
    - Preferred creators: Andrew Ng, Andrej Karpathy

    Recommended article:
    - Title: "Advanced Transformer Architectures"
    - Summary: Deep dive into attention mechanisms and modern transformer designs
    - Creator: Yan LeCun

    Provide a 1-2 sentence explanation starting with "Because..."
    """

    print("Generating explanation...")

    explanation = await llm.agenerate(explain_prompt, max_tokens=100)

    print(f"\n✓ Explanation:\n{explanation}\n")

    print("=" * 60)
    print("✅ ALL REAL LLM TESTS PASSED!")
    print("\n🎉 The system works with actual API calls!\n")


if __name__ == "__main__":
    asyncio.run(test_real_llm())
