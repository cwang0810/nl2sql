"""Base class for SQL generators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..models.base import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class SQLCandidate:
    """A generated SQL candidate with metadata."""
    sql: str
    generator: str               # which generator produced this
    model: str                   # which LLM model
    temperature: float           # sampling temperature
    prompt_strategy: str         # prompt template used
    raw_response: str = ""       # full LLM response
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.sql)

    def __eq__(self, other):
        return isinstance(other, SQLCandidate) and self.sql == other.sql


class SQLGenerator(ABC):
    """Abstract base class for SQL generators."""

    def __init__(self, llm_client: LLMClient, name: str = ""):
        self.llm = llm_client
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def generate(
        self,
        question: str,
        schema: str,
        evidence: str = "",
        few_shot_examples: str = "",
        retrieved_values: str = "",
        **kwargs: Any,
    ) -> list[SQLCandidate]:
        """Generate SQL candidates for a question.

        Args:
            question: Natural language question.
            schema: Formatted database schema.
            evidence: External knowledge / evidence.
            few_shot_examples: Few-shot demonstration examples.
            retrieved_values: Retrieved cell values.

        Returns:
            List of SQLCandidate objects.
        """
        ...
