"""
SQL Fixer: fix syntax errors using LLM with execution feedback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..models.base import LLMClient
from ..utils.db_executor import ExecutionResult, execute_sql

logger = logging.getLogger(__name__)

FIX_PROMPT = """You are an expert SQL debugger. The following SQL query has an execution error. Fix it.

【Database Schema】
{schema}

【Question】
{question}

【Original SQL】
{sql}

【Error Message】
{error}

Fix the SQL query. Output ONLY the corrected SQL, nothing else:"""


class SQLFixer:
    """Fix SQL syntax errors using execution feedback and LLM correction."""

    def __init__(self, llm_client: LLMClient, max_retries: int = 3):
        self.llm = llm_client
        self.max_retries = max_retries

    async def fix(
        self,
        sql: str,
        db_path: str | Path,
        db_id: str,
        question: str,
        schema: str,
        timeout: int = 30,
    ) -> tuple[str, bool]:
        """Attempt to fix a SQL query that produces execution errors.

        Args:
            sql: Original SQL query.
            db_path: Path to SQLite database.
            db_id: Database identifier.
            question: Original natural language question.
            schema: Database schema string.
            timeout: SQL execution timeout.

        Returns:
            Tuple of (fixed_sql, was_fixed).
        """
        current_sql = sql

        for attempt in range(self.max_retries):
            result = execute_sql(current_sql, db_path, db_id, timeout)

            if result.success:
                if attempt > 0:
                    logger.info(f"SQL fixed after {attempt} attempt(s)")
                return current_sql, attempt > 0

            # Ask LLM to fix the error
            prompt = FIX_PROMPT.format(
                schema=schema,
                question=question,
                sql=current_sql,
                error=result.error,
            )

            try:
                responses = await self.llm.generate(prompt, temperature=0.0)
                new_sql = responses[0].sql
                if new_sql and new_sql != current_sql:
                    current_sql = new_sql
                else:
                    break  # LLM couldn't produce a different fix
            except Exception as e:
                logger.warning(f"SQL fixer LLM call failed: {e}")
                break

        return current_sql, current_sql != sql
