"""
Self-consistency selection: majority voting based on execution results.
Used as fallback when tournament selection produces ties.
"""

from __future__ import annotations

import logging
from collections import Counter

from ..stage2_generation.generator_base import SQLCandidate
from ..utils.db_executor import ExecutionResult

logger = logging.getLogger(__name__)


class SelfConsistencySelector:
    """Select SQL by majority voting on execution results."""

    def select(
        self,
        candidates: list[tuple[SQLCandidate, ExecutionResult]],
    ) -> SQLCandidate:
        """Select the candidate whose execution result appears most frequently.

        Args:
            candidates: List of (candidate, execution_result) tuples.

        Returns:
            The candidate with the most common result.
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            return candidates[0][0]

        # Group by result signature
        result_counts: Counter = Counter()
        result_map: dict[str, list[tuple[SQLCandidate, ExecutionResult]]] = {}

        for candidate, result in candidates:
            sig = result.result_signature()
            result_counts[sig] += 1
            result_map.setdefault(sig, []).append((candidate, result))

        # Pick the most common result
        most_common_sig = result_counts.most_common(1)[0][0]
        group = result_map[most_common_sig]

        # From the winning group, pick the shortest SQL
        group.sort(key=lambda x: len(x[0].sql))
        winner = group[0][0]

        logger.info(
            f"Self-consistency: selected result appearing "
            f"{result_counts[most_common_sig]}/{len(candidates)} times, "
            f"generator={winner.generator}"
        )
        return winner
