"""
SQL Revisor: semantic review and correction of SQL queries.
Handles cases where SQL executes successfully but may be logically wrong.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..models.base import LLMClient
from ..utils.db_executor import ExecutionResult, execute_sql

logger = logging.getLogger(__name__)

REVISE_PROMPT = """You are an expert SQL reviewer. A SQL query was generated for the following question and executed successfully, but the result may be incorrect (empty or suspicious). Review and fix it.

【Database Schema】
{schema}

【Question】
{question}

【External Knowledge】
{evidence}

【Generated SQL】
{sql}

【Execution Result】
Columns: {columns}
Rows (first 10): {rows}
Total rows: {total_rows}

Common issues to check:
1. **Empty result?** Check if the query uses the WRONG TABLE for date filtering. Look at the schema — some tables only have data for limited date ranges. If the question asks about a date outside that range, you must use a different table.
2. Are the JOIN conditions correct?
3. Are the WHERE conditions accurate? Check exact value casing (e.g., 'Discount' vs 'discount').
4. Are the aggregation functions appropriate?
5. Does the SQL match the evidence formula EXACTLY?

If the SQL is correct, output it unchanged. If it needs fixing, output the corrected SQL.
Output ONLY the SQL query:"""


class SQLRevisor:
    """Semantic review and correction of SQL queries."""

    def __init__(self, llm_client: LLMClient, max_retries: int = 2):
        self.llm = llm_client
        self.max_retries = max_retries

    async def revise(
        self,
        sql: str,
        result: ExecutionResult,
        question: str,
        schema: str,
        evidence: str = "",
        db_path: str | Path = "",
        db_id: str = "",
        timeout: int = 30,
    ) -> tuple[str, bool]:
        """Review and potentially revise a SQL query.

        Args:
            sql: Current SQL query.
            result: Execution result of the query.
            question: Original question.
            schema: Database schema.
            evidence: External knowledge.
            db_path: Database path (for re-execution).
            db_id: Database identifier.
            timeout: SQL execution timeout.

        Returns:
            Tuple of (revised_sql, was_revised).
        """
        current_sql = sql
        current_result = result

        for attempt in range(self.max_retries):
            # Format result for prompt
            rows_str = str(current_result.rows[:10])
            columns_str = str(current_result.columns)

            prompt = REVISE_PROMPT.format(
                schema=schema,
                question=question,
                evidence=evidence or "No additional knowledge.",
                sql=current_sql,
                columns=columns_str,
                rows=rows_str,
                total_rows=len(current_result.rows),
            )

            try:
                responses = await self.llm.generate(prompt, temperature=0.0)
                new_sql = responses[0].sql

                if not new_sql or new_sql == current_sql:
                    break  # LLM thinks it's correct or couldn't improve

                # Verify the revised SQL executes successfully
                if db_path:
                    new_result = execute_sql(new_sql, db_path, db_id, timeout)
                    if new_result.success:
                        current_sql = new_sql
                        current_result = new_result
                    else:
                        break  # Revision broke the query
                else:
                    current_sql = new_sql
                    break

            except Exception as e:
                logger.warning(f"SQL revisor LLM call failed: {e}")
                break

        return current_sql, current_sql != sql

    def should_revise(self, result: ExecutionResult) -> bool:
        """Determine if a SQL result should be reviewed.

        Only triggers for empty results — the most reliable signal.
        Suspiciously large results are no longer revised since the revisor
        was found to degrade accuracy on non-empty results in evaluation.
        """
        return result.is_empty
