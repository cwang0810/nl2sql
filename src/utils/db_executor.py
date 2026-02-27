"""SQL executor with timeout control for SQLite databases."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from func_timeout import FunctionTimedOut, func_timeout

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a SQL execution."""
    sql: str
    db_id: str
    success: bool
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    execution_time_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.success and len(self.rows) == 0

    @property
    def is_error(self) -> bool:
        return not self.success

    @property
    def is_suspicious(self) -> bool:
        """Heuristic: result might be wrong if it returns too many rows."""
        return self.success and len(self.rows) > 10000

    @property
    def result_set(self) -> set[tuple]:
        """Return result as a set of tuples for comparison."""
        return set(self.rows)

    def result_signature(self) -> str:
        """A hashable signature of the result for deduplication."""
        if not self.success:
            return f"ERROR:{self.error}"
        # Use None-safe sorting: replace None with a sentinel for comparison
        def _sort_key(row: tuple) -> tuple:
            return tuple(
                (0, "") if v is None else (1, v) if isinstance(v, str) else (2, v)
                for v in row
            )
        try:
            return str(sorted(self.rows, key=_sort_key))
        except TypeError:
            # Final fallback: convert to strings for sorting
            return str(sorted(self.rows, key=lambda r: str(r)))


def _execute_sql_inner(db_path: str, sql: str) -> tuple[list[str], list[tuple]]:
    """Execute SQL on a SQLite database (inner function for timeout)."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, rows
    finally:
        conn.close()


def execute_sql(
    sql: str,
    db_path: str | Path,
    db_id: str = "",
    timeout_seconds: int = 30,
) -> ExecutionResult:
    """Execute a SQL query against a SQLite database with timeout.

    Args:
        sql: SQL query string.
        db_path: Path to the SQLite database file.
        db_id: Database identifier (for logging).
        timeout_seconds: Maximum execution time in seconds.

    Returns:
        ExecutionResult with success/failure info and result rows.
    """
    db_path = str(db_path)
    start = time.monotonic()

    try:
        columns, rows = func_timeout(
            timeout_seconds,
            _execute_sql_inner,
            args=(db_path, sql),
        )
        elapsed = (time.monotonic() - start) * 1000
        return ExecutionResult(
            sql=sql,
            db_id=db_id,
            success=True,
            rows=rows,
            columns=columns,
            execution_time_ms=elapsed,
        )
    except FunctionTimedOut:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning(f"SQL execution timed out after {timeout_seconds}s: {sql[:100]}...")
        return ExecutionResult(
            sql=sql,
            db_id=db_id,
            success=False,
            error="timeout",
            execution_time_ms=elapsed,
        )
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        logger.debug(f"SQL execution error: {e}")
        return ExecutionResult(
            sql=sql,
            db_id=db_id,
            success=False,
            error=str(e),
            execution_time_ms=elapsed,
        )


def get_db_path(db_root: str | Path, db_id: str) -> Path:
    """Resolve the SQLite database file path from db_root and db_id.

    BIRD convention: {db_root}/{db_id}/{db_id}.sqlite
    """
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def get_table_names(db_path: str | Path) -> list[str]:
    """Get all table names from a SQLite database."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def get_table_schema(db_path: str | Path, table_name: str) -> list[dict[str, Any]]:
    """Get schema info (columns, types) for a table."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "cid": row[0],
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "default": row[4],
                "pk": bool(row[5]),
            })
        return columns
    finally:
        conn.close()


def get_foreign_keys(db_path: str | Path, table_name: str) -> list[dict[str, str]]:
    """Get foreign key relationships for a table."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
        fks = []
        for row in cursor.fetchall():
            fks.append({
                "from_column": row[3],
                "to_table": row[2],
                "to_column": row[4],
            })
        return fks
    finally:
        conn.close()


def get_sample_values(
    db_path: str | Path,
    table_name: str,
    column_name: str,
    limit: int = 5,
) -> list:
    """Get sample distinct values from a column."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT DISTINCT `{column_name}` FROM `{table_name}` "
            f"WHERE `{column_name}` IS NOT NULL LIMIT ?;",
            (limit,),
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()
