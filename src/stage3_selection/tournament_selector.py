"""
Tournament Selection: position-debiased pairwise LLM comparison to select the best SQL.
Inspired by Agentar-Scale-SQL with position bias mitigation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..models.base import LLMClient
from ..stage2_generation.generator_base import SQLCandidate
from ..utils.db_executor import ExecutionResult

logger = logging.getLogger(__name__)

PAIRWISE_PROMPT = """You are an expert SQL judge for SQLite databases. Given a natural language question, database schema, and external knowledge, compare two SQL queries and determine which is more likely to produce the CORRECT answer.

【Question】
{question}

【Database Schema】
{schema}

【External Knowledge / Evidence】
{evidence}

【SQL Candidate 1】
```sql
{sql_a}
```
Execution result (first 5 rows): {result_a}
Row count: {count_a}

【SQL Candidate 2】
```sql
{sql_b}
```
Execution result (first 5 rows): {result_b}
Row count: {count_b}

CRITICAL analysis checklist:
1. **Evidence compliance**: If evidence provides a formula or mapping, which SQL follows it EXACTLY?
2. **Column selection**: Which SQL returns ONLY the columns the question asks for? (Extra columns = wrong)
3. **Column order**: Do the SELECT columns follow the order mentioned in the question?
4. **JOIN correctness**: Are the JOIN conditions and table relationships correct?
5. **WHERE accuracy**: Do the filter conditions match the question's requirements?
6. **Aggregation scope**: Is COUNT/SUM/AVG applied at the correct level (per-row vs per-group vs overall)?
7. **Result reasonableness**: Which result makes more sense given the question? Empty results or NULL-heavy results are suspicious.
8. **SQLite syntax**: Is the SQL valid SQLite? (SUBSTR not SUBSTRING, || for concat, no ISNULL)

Based on this analysis, which SQL is more likely correct?
You MUST choose one. Output ONLY "1" or "2" (the candidate number):"""


class TournamentSelector:
    """Position-debiased tournament-style pairwise comparison for SQL selection."""

    def __init__(self, llm_client: LLMClient, temperature: float = 0.0):
        self.llm = llm_client
        self.temperature = temperature

    async def _pairwise_compare_single(
        self,
        question: str,
        schema: str,
        evidence: str,
        candidate_a: SQLCandidate,
        result_a: ExecutionResult,
        candidate_b: SQLCandidate,
        result_b: ExecutionResult,
    ) -> str:
        """Single directional comparison, return "1" or "2"."""
        prompt = PAIRWISE_PROMPT.format(
            question=question,
            schema=schema,
            evidence=evidence or "No additional knowledge.",
            sql_a=candidate_a.sql,
            result_a=str(result_a.rows[:5]) if result_a.success else f"Error: {result_a.error}",
            count_a=len(result_a.rows) if result_a.success else "N/A",
            sql_b=candidate_b.sql,
            result_b=str(result_b.rows[:5]) if result_b.success else f"Error: {result_b.error}",
            count_b=len(result_b.rows) if result_b.success else "N/A",
        )

        try:
            responses = await self.llm.generate(
                prompt=prompt,
                temperature=self.temperature,
            )
            answer = responses[0].content.strip()

            if "1" in answer and "2" not in answer:
                return "1"
            elif "2" in answer and "1" not in answer:
                return "2"
            else:
                # Ambiguous — check first non-whitespace char
                for ch in answer:
                    if ch == "1":
                        return "1"
                    if ch == "2":
                        return "2"
                return "1"  # Default fallback
        except Exception as e:
            logger.warning(f"Pairwise comparison failed: {e}")
            return "1"

    async def _pairwise_compare_debiased(
        self,
        question: str,
        schema: str,
        evidence: str,
        candidate_a: SQLCandidate,
        result_a: ExecutionResult,
        candidate_b: SQLCandidate,
        result_b: ExecutionResult,
    ) -> str:
        """Position-debiased comparison: run both orderings and require consistency.

        Returns "A" if candidate_a wins, "B" if candidate_b wins,
        or "TIE" if results are inconsistent (position bias detected).
        """
        # Forward: A=1, B=2
        forward_task = self._pairwise_compare_single(
            question, schema, evidence,
            candidate_a, result_a,
            candidate_b, result_b,
        )
        # Reverse: B=1, A=2
        reverse_task = self._pairwise_compare_single(
            question, schema, evidence,
            candidate_b, result_b,
            candidate_a, result_a,
        )

        forward_result, reverse_result = await asyncio.gather(forward_task, reverse_task)

        # Forward: "1" means A wins, "2" means B wins
        # Reverse: "1" means B wins, "2" means A wins
        a_wins_forward = (forward_result == "1")
        a_wins_reverse = (reverse_result == "2")

        if a_wins_forward and a_wins_reverse:
            return "A"  # Consistent: A wins both ways
        elif not a_wins_forward and not a_wins_reverse:
            return "B"  # Consistent: B wins both ways
        else:
            return "TIE"  # Inconsistent: position bias detected

    async def select(
        self,
        candidates: list[tuple[SQLCandidate, ExecutionResult]],
        question: str,
        schema: str,
        evidence: str = "",
    ) -> SQLCandidate:
        """Select the best SQL using position-debiased tournament.

        Args:
            candidates: List of (candidate, execution_result) tuples,
                        ordered by group size (most common first).
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

        # Round-robin tournament with position debiasing
        n = len(candidates)
        wins = {i: 0 for i in range(n)}
        ties = {i: 0 for i in range(n)}

        # Create all pairwise comparison tasks
        comparison_tasks = []
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                task = self._pairwise_compare_debiased(
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
            elif winner == "B":
                wins[j] += 1
            else:
                # TIE: give half credit to the candidate from a larger group
                # (deduplicator orders by group_size descending)
                ties[i] += 1
                ties[j] += 1

        # Score = consistent wins + 0.25 * ties (slight credit for ties)
        # Tiebreaker: prefer candidates from larger groups (lower index = larger group)
        scores = {i: wins[i] + 0.25 * ties[i] for i in range(n)}
        best_idx = max(range(n), key=lambda i: (scores[i], -i))
        winner_candidate = candidates[best_idx][0]

        logger.info(
            f"Tournament (debiased): {n} candidates, "
            f"winner idx={best_idx} with {wins[best_idx]} wins + {ties[best_idx]} ties, "
            f"generator={winner_candidate.generator}, strategy={winner_candidate.prompt_strategy}"
        )
        return winner_candidate
