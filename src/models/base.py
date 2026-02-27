"""
LLM Client base class - unified interface for all LLM API calls.
Supports OpenAI-compatible APIs (DeepSeek, Qwen3, etc.)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import openai

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def sql(self) -> str:
        """Extract SQL from response content."""
        return extract_sql(self.content)


def extract_sql(text: str) -> str:
    """Extract SQL query from LLM response text.

    Handles multiple formats:
    - ```sql ... ```
    - ```SQL ... ```
    - ``` ... ```
    - Raw SQL text
    """
    # Try code block extraction first
    patterns = [
        r"```sql\s*(.*?)```",
        r"```SQL\s*(.*?)```",
        r"```\s*(SELECT.*?)```",
        r"```\s*(WITH.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Try to find raw SQL starting with SELECT or WITH
    match = re.search(
        r"((?:SELECT|WITH)\s+.*?)(?:\n\n|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().rstrip(";")

    # Fallback: return the whole text stripped
    return text.strip().rstrip(";")


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a SQL expert.",
        temperature: float = 0.0,
        n: int = 1,
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """Generate responses from the LLM.

        Args:
            prompt: User prompt.
            system_prompt: System instruction.
            temperature: Sampling temperature.
            n: Number of responses to generate.

        Returns:
            List of LLMResponse objects.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(self.max_retries):
            try:
                start = time.monotonic()
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    n=n,
                    **kwargs,
                )
                latency = (time.monotonic() - start) * 1000

                results = []
                for choice in response.choices:
                    results.append(LLMResponse(
                        content=choice.message.content or "",
                        model=self.model_name,
                        usage={
                            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        },
                        latency_ms=latency,
                    ))
                return results

            except (openai.RateLimitError, openai.APITimeoutError) as e:
                wait = 2 ** attempt
                logger.warning(
                    f"API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            except openai.APIError as e:
                logger.error(f"API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Failed after {self.max_retries} retries")

    async def generate_sql(
        self,
        prompt: str,
        system_prompt: str = "You are a SQL expert. Output only the SQL query.",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Generate a single SQL query.

        Convenience method that extracts SQL from the response.
        """
        responses = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            n=1,
            **kwargs,
        )
        return responses[0].sql

    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...
