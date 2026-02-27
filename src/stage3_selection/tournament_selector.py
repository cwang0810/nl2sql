"""
Tournament Selection: pairwise LLM comparison to select the best SQL.
Inspired by Agentar-Scale-SQL.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..models.base import LLMClient
from ..stage2_generation.generator_base import SQLCandidate
from ..utils.db_executor import ExecutionResult

logger = logging.getLogger(__name__)

PAIRWISE_PROMPT = """You are an expert SQL judge. Given a natural language question and database schema, compare two SQL queries and determine which is more likely correct.

【Question】
{question}

【Database Schema】
{schema}

【External Knowledge】
{evidence}

【SQL A】
{sql_a}
Execution result (first 5 rows): {result_a}

【SQL B】
{sql_b}
Execution result (first 5 rows): {result_b}

Analyze both SQL queries:
1. Does each correctly interpret the question's intent?
2. Are JOIN conditions correct?
3. Are WHERE conditions accurate?
4. Are aggregation functions appropriate?
5. Which result looks more reasonable?

You MUST choose one. Output ONLY "A" or "B":"""


class TournamentSelector:
    """Tournament-style pairwise comparison for SQL selection."""

    def __init__(self, llm_client: LLMClient, temperature: float = 0.0):
        self.llm = llm_client
        self.temperature = temperature

    async def _pairwise_compare(
        self,
        question: str,
        schema: str,
        evidence: str,
        candidate_a: SQLCandidate,
        result_a: ExecutionResult,
        candidate_b: SQLCandidate,
        result_b: ExecutionResult,
    ) -> str:
        """Compare two candidates, return "A" or "B"."""
        prompt = PAIRWISE_PROMPT.format(
            question=question,
            schema=schema,
            evidence=evidence or "No additional knowledge.",
            sql_a=candidate_a.sql,
            result_a=str(result_a.rows[:5]) if result_a.success else f"Error: {result_a.error}",
            sql_b=candidate_b.sql,
            result_b=str(result_b.rows[:5]) if result_b.success else f"Error: {result_b.error}",
        )

        try:
            responses = await self.llm.generate(
                prompt=prompt,
                temperature=self.temperature,
            )
            answer = responses[0].content.strip().upper()

            if "A" in answer and "B" not in answer:
                return "A"
            elif "B" in answer and "A" not in answer:
                return "B"
            else:
                # Ambiguous answer, prefer A (first candidate)
                return "A" if answer.startswith("A") else "B"
        except Exception as e:
            logger.warning(f"Pairwise comparison failed: {e}")
            return "A"  # Default to first candidate on failure

    async def select(
        self,
        candidates: list[tuple[SQLCandidate, ExecutionResult]],
        question: str,
        schema: str,
        evidence: str = "",
    ) -> SQLCandidate:
        """Select the best SQL using tournament-style pairwise comparison.

        Args:
            candidates: List of (candidate, execution_result) tuples.
            question: Original natural language question.
            schema: Database schema string.
            evidence: External knowledge.

        Returns:
            The winning SQLCandidate.
        """
        if len(candidates) == 0:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            return candidates[0][0]

        # Round-robin tournament
        wins = {i: 0 for i in range(len(candidates))}

        # Create all pairwise comparison tasks
        comparison_tasks = []
        pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                task = self._pairwise_compare(
                    question=question,
                    schema=schema,
                    evidence=evidence,
                    candidate_a=candidates[i][0],
                    result_a=candidates[i][1],
                    candidate_b=candidates[j][0],
                    result_b=candidates[j][1],
                )
                comparison_tasks.append(task)
                pairs.append((i, j))

        # Run all comparisons concurrently
        results = await asyncio.gather(*comparison_tasks)

        for (i, j), winner in zip(pairs, results):
            if winner == "A":
                wins[i] += 1
            else:
                wins[j] += 1

        # Select the candidate with most wins
        best_idx = max(wins, key=wins.get)
        winner = candidates[best_idx][0]

        logger.info(
            f"Tournament selection: {len(candidates)} candidates, "
            f"winner has {wins[best_idx]} wins, "
            f"generator={winner.generator}, strategy={winner.prompt_strategy}"
        )
        return winner
