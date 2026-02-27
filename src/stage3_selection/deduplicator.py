"""
Execution-based deduplication: group SQL candidates by their execution results.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from ..stage2_generation.generator_base import SQLCandidate
from ..utils.db_executor import ExecutionResult, execute_sql

logger = logging.getLogger(__name__)


class Deduplicator:
    """Deduplicate SQL candidates by execution results."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def deduplicate(
        self,
        candidates: list[SQLCandidate],
        db_path: str | Path,
        db_id: str = "",
    ) -> list[tuple[SQLCandidate, ExecutionResult]]:
        """Deduplicate candidates by grouping on execution results.

        For each group of candidates producing the same result,
        keep the shortest SQL as representative.

        Args:
            candidates: List of SQL candidates.
            db_path: Path to SQLite database.
            db_id: Database identifier.

        Returns:
            List of (representative_candidate, execution_result) tuples,
            ordered by group size (most common result first).
        """
        groups: dict[str, list[tuple[SQLCandidate, ExecutionResult]]] = defaultdict(list)

        for candidate in candidates:
            result = execute_sql(candidate.sql, db_path, db_id, self.timeout)
            sig = result.result_signature()
            groups[sig].append((candidate, result))

        # Sort groups by size (most common first)
        sorted_groups = sorted(groups.values(), key=len, reverse=True)

        representatives = []
        for group in sorted_groups:
            # Pick shortest SQL as representative
            group.sort(key=lambda x: len(x[0].sql))
            rep_candidate, rep_result = group[0]

            # Attach group size as metadata
            rep_candidate.metadata["group_size"] = len(group)
            rep_candidate.metadata["group_generators"] = list({c.generator for c, _ in group})

            representatives.append((rep_candidate, rep_result))

        logger.info(
            f"Deduplicated {len(candidates)} candidates into "
            f"{len(representatives)} unique result groups"
        )
        return representatives
