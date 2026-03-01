"""
Self-consistency selection: weighted majority voting based on execution results.
Weights are based on empirical performance of different generators and strategies.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from ..stage2_generation.generator_base import SQLCandidate
from ..utils.db_executor import ExecutionResult

logger = logging.getLogger(__name__)

# Empirical weights based on evaluation results:
# - ICL direct (70.3%) > divide_conquer (59.5%) > CoT (57.1%) > execution_plan (57.0%)
# - DeepSeek (66.6%) > Qwen3 (57.7%)
# - Low temperature is generally more reliable
GENERATOR_WEIGHTS = {
    "ICLGenerator": 1.2,
    "CoTGenerator": 1.0,
    "ICLGenerator+fixer": 0.3,    # Fixer degrades quality
    "CoTGenerator+fixer": 0.2,
    "ICLGenerator+revisor": 0.6,
    "CoTGenerator+revisor": 0.3,
}

STRATEGY_WEIGHTS = {
    "direct": 1.3,
    "icl_direct": 1.3,
    "cot": 1.0,
    "icl_cot": 1.0,
    "execution_plan": 1.0,
    "cot_execution_plan": 1.0,
    "divide_conquer": 1.05,
}

TEMP_WEIGHTS = {
    0.0: 1.2,
    0.5: 1.1,
    1.0: 1.0,
    1.5: 0.9,
}


def _candidate_weight(candidate: SQLCandidate) -> float:
    """Compute the voting weight for a candidate based on its metadata."""
    gen_w = GENERATOR_WEIGHTS.get(candidate.generator, 1.0)
    strat_w = STRATEGY_WEIGHTS.get(candidate.prompt_strategy, 1.0)
    temp_w = TEMP_WEIGHTS.get(candidate.temperature, 1.0)
    return gen_w * strat_w * temp_w


class SelfConsistencySelector:
    """Select SQL by weighted majority voting on execution results."""

    def select(
        self,
        candidates: list[tuple[SQLCandidate, ExecutionResult]],
    ) -> SQLCandidate:
        """Select the candidate whose execution result has highest weighted votes.

        Args:
            candidates: List of (candidate, execution_result) tuples.

        Returns:
            The candidate with the highest weighted vote score.
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            return candidates[0][0]

        # Group by result signature with weighted votes
        result_weights: dict[str, float] = defaultdict(float)
        result_counts: Counter = Counter()
        result_map: dict[str, list[tuple[SQLCandidate, ExecutionResult]]] = {}

        for candidate, result in candidates:
            sig = result.result_signature()
            weight = _candidate_weight(candidate)
            result_weights[sig] += weight
            result_counts[sig] += 1
            result_map.setdefault(sig, []).append((candidate, result))

        # Pick the result with highest weighted score
        best_sig = max(result_weights, key=result_weights.get)
        group = result_map[best_sig]

        # From the winning group, pick the candidate with highest individual weight,
        # then shortest SQL as tiebreaker
        group.sort(key=lambda x: (-_candidate_weight(x[0]), len(x[0].sql)))
        winner = group[0][0]

        logger.info(
            f"Self-consistency (weighted): selected result with "
            f"weighted_score={result_weights[best_sig]:.2f}, "
            f"count={result_counts[best_sig]}/{len(candidates)}, "
            f"generator={winner.generator}"
        )
        return winner
