"""
Schema formatting utilities.

Produces two schema representations:
- Light Schema (Markdown) - concise, good for ICL prompts
- DDL Schema (CREATE TABLE) - detailed, good for CoT reasoning

Enhanced with table statistics (row count, date ranges, distinct value counts)
to help models understand the data distribution.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from ..utils.db_executor import (
    get_foreign_keys,
    get_sample_values,
    get_table_names,
    get_table_schema,
)


def load_database_descriptions(db_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Load BIRD database_description CSV files.

    BIRD convention: {db_dir}/database_description/{table_name}.csv
    Each CSV has columns: original_column_name, column_name, column_description, data_format, value_description

    Returns:
        Dict mapping table_name -> list of column description dicts.
    """
    desc_dir = db_dir / "database_description"
    descriptions: dict[str, list[dict[str, str]]] = {}

    if not desc_dir.exists():
        return descriptions

    for csv_file in desc_dir.glob("*.csv"):
        table_name = csv_file.stem
        columns = []
        try:
            with open(csv_file, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    columns.append({
                        "original_column_name": row.get("original_column_name", "").strip(),
                        "column_name": row.get("column_name", "").strip(),
                        "column_description": row.get("column_description", "").strip(),
                        "data_format": row.get("data_format", "").strip(),
                        "value_description": row.get("value_description", "").strip(),
                    })
        except Exception:
            pass
        descriptions[table_name] = columns

    return descriptions


def _get_domain_hints(db_name: str, comment_prefix: str = "") -> str:
    """Load database-specific domain hints from config/domain_hints.yaml."""
    from ..utils.prompt_loader import load_domain_hints
    return load_domain_hints(db_name, comment_prefix=comment_prefix)


def _get_table_stats(db_path: Path, table: str) -> dict[str, Any]:
    """Get table statistics: row count, date/time column ranges, low-cardinality enums."""
    stats: dict[str, Any] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        c = conn.cursor()
        # Row count
        c.execute(f"SELECT COUNT(*) FROM `{table}`")
        stats["row_count"] = c.fetchone()[0]

        # Per-column stats
        c.execute(f"PRAGMA table_info(`{table}`)")
        columns = c.fetchall()

        col_stats: dict[str, dict] = {}
        for col_info in columns:
            col_name = col_info[1]
            col_type = (col_info[2] or "").upper()

            # Check distinct count
            try:
                c.execute(
                    f"SELECT COUNT(DISTINCT `{col_name}`) FROM `{table}` "
                    f"WHERE `{col_name}` IS NOT NULL"
                )
                distinct_count = c.fetchone()[0]
            except Exception:
                continue

            cs: dict[str, Any] = {"distinct_count": distinct_count}

            # For low-cardinality columns (<=20 distinct), list all values
            if 0 < distinct_count <= 20:
                try:
                    c.execute(
                        f"SELECT DISTINCT `{col_name}` FROM `{table}` "
                        f"WHERE `{col_name}` IS NOT NULL "
                        f"ORDER BY `{col_name}` LIMIT 20"
                    )
                    cs["all_values"] = [r[0] for r in c.fetchall()]
                except Exception:
                    pass

            # For date/time-like columns, get min/max
            if any(kw in col_type for kw in ("DATE", "TIME", "YEAR")) or \
               any(kw in col_name.lower() for kw in ("date", "time", "year", "month", "day")):
                try:
                    c.execute(
                        f"SELECT MIN(`{col_name}`), MAX(`{col_name}`) FROM `{table}` "
                        f"WHERE `{col_name}` IS NOT NULL"
                    )
                    row = c.fetchone()
                    if row and row[0] is not None:
                        cs["min"] = row[0]
                        cs["max"] = row[1]
                except Exception:
                    pass

            if len(cs) > 1:  # has more than just distinct_count
                col_stats[col_name] = cs

        stats["columns"] = col_stats
    except Exception:
        pass
    finally:
        conn.close()
    return stats


def _format_table_stats(table: str, stats: dict[str, Any]) -> str:
    """Format table statistics as a compact string."""
    parts = [f"Rows: {stats.get('row_count', '?')}"]

    col_stats = stats.get("columns", {})
    for col_name, cs in col_stats.items():
        if "min" in cs and "max" in cs:
            parts.append(f"{col_name} range: [{cs['min']}..{cs['max']}]")
        if "all_values" in cs:
            vals = cs["all_values"]
            vals_str = ", ".join(repr(v) for v in vals)
            parts.append(f"{col_name} values({len(vals)}): {vals_str}")

    return "; ".join(parts)


def format_light_schema(
    db_path: str | Path,
    db_dir: Path | None = None,
    relevant_tables: list[str] | None = None,
    include_samples: bool = True,
    max_samples: int = 3,
) -> str:
    """Format database schema as Light Schema (Markdown).

    Example output:
    ## Table: employees (1000 rows)
    | Column | Type | Description | Examples |
    |--------|------|-------------|----------|
    | id | INTEGER | Primary key |
    | name | TEXT | Employee name | Examples: "Alice", "Bob" |
    ...
    Foreign Keys: employees.dept_id → departments.id
    Stats: date range: [2012-01-01..2013-12-31]; status values(3): 'A', 'B', 'C'
    """
    db_path = Path(db_path)
    tables = relevant_tables or get_table_names(db_path)
    descriptions = load_database_descriptions(db_dir) if db_dir else {}

    # Build overview of all tables first, collecting limited-range warnings
    all_stats = {}
    overview_warnings = []
    for table in tables:
        if table == "sqlite_sequence":
            continue
        stats = _get_table_stats(db_path, table)
        all_stats[table] = stats
        row_count = stats.get("row_count", "?")
        for col_name, cs in stats.get("columns", {}).items():
            if "all_values" in cs and "min" in cs and len(cs["all_values"]) <= 10:
                col_lower = col_name.lower()
                if any(kw in col_lower for kw in ("date", "time", "year", "month")):
                    vals_str = ", ".join(repr(v) for v in cs["all_values"])
                    overview_warnings.append(
                        f"- ⚠️ {table} ({row_count} rows): {col_name} ONLY contains "
                        f"{vals_str}. Queries filtering by other {col_name} values will return EMPTY results!"
                    )

    parts = []
    if overview_warnings:
        parts.append(
            "# ⚠️ IMPORTANT DATA LIMITATIONS\n"
            + "\n".join(overview_warnings)
            + "\nWhen a question references dates outside these ranges, use a DIFFERENT table!\n"
        )

    # Add database-specific domain hints
    db_name = db_path.parent.name if db_path.parent else ""
    domain_hints = _get_domain_hints(db_name)
    if domain_hints:
        parts.append(domain_hints)

    for table in tables:
        if table == "sqlite_sequence":
            continue

        columns = get_table_schema(db_path, table)
        fks = get_foreign_keys(db_path, table)
        table_desc = descriptions.get(table, [])
        desc_map = {d["original_column_name"]: d for d in table_desc}

        # Get table stats (use cached if available)
        stats = all_stats.get(table) or _get_table_stats(db_path, table)
        row_count = stats.get("row_count", "?")

        lines = [f"## Table: {table} ({row_count} rows)"]
        lines.append("| Column | Type | Description | Examples |")
        lines.append("|--------|------|-------------|----------|")

        for col in columns:
            col_name = col["name"]
            col_type = col["type"] or "TEXT"
            pk_mark = " (PK)" if col["pk"] else ""

            desc = ""
            if col_name in desc_map:
                d = desc_map[col_name]
                desc = d.get("column_description", "")
                if d.get("value_description"):
                    desc += f" [{d['value_description']}]"

            # Use all_values from stats if available (low cardinality), else samples
            samples = ""
            col_stat = stats.get("columns", {}).get(col_name, {})
            if "all_values" in col_stat:
                vals = col_stat["all_values"]
                samples = ", ".join(repr(v) for v in vals[:10])
                if len(vals) > 10:
                    samples += f" ... ({len(vals)} total)"
            elif include_samples:
                try:
                    vals = get_sample_values(db_path, table, col_name, max_samples)
                    if vals:
                        samples = ", ".join(repr(v) for v in vals)
                except Exception:
                    pass

            lines.append(f"| {col_name}{pk_mark} | {col_type} | {desc} | {samples} |")

        if fks:
            fk_strs = [f"{table}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}" for fk in fks]
            lines.append(f"Foreign Keys: {'; '.join(fk_strs)}")

        # Add date/time range info and warnings
        stat_notes = []
        warnings = []
        for col_name, cs in stats.get("columns", {}).items():
            if "min" in cs and "max" in cs:
                min_val, max_val = str(cs['min']), str(cs['max'])
                stat_notes.append(f"{col_name} range: [{min_val} .. {max_val}]")
                # Check for very limited date ranges (less than 30 distinct values)
                if "all_values" in cs and len(cs["all_values"]) <= 10:
                    vals_str = ", ".join(repr(v) for v in cs["all_values"])
                    warnings.append(
                        f"⚠️ WARNING: {table}.{col_name} ONLY has these {len(cs['all_values'])} values: "
                        f"{vals_str}. Do NOT filter by {col_name} values outside this set — "
                        f"use a different table instead!"
                    )
        if stat_notes:
            lines.append(f"Data ranges: {'; '.join(stat_notes)}")
        for w in warnings:
            lines.append(w)

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def format_ddl_schema(
    db_path: str | Path,
    db_dir: Path | None = None,
    relevant_tables: list[str] | None = None,
    include_comments: bool = True,
) -> str:
    """Format database schema as DDL (CREATE TABLE statements).

    Enhanced with table statistics as comments.
    """
    db_path = Path(db_path)
    tables = relevant_tables or get_table_names(db_path)
    descriptions = load_database_descriptions(db_dir) if db_dir else {}

    # Collect limited-range warnings for DDL overview
    overview_warnings = []
    all_stats_ddl = {}
    for table in tables:
        if table == "sqlite_sequence":
            continue
        stats = _get_table_stats(db_path, table)
        all_stats_ddl[table] = stats
        row_count = stats.get("row_count", "?")
        for col_name, cs in stats.get("columns", {}).items():
            if "all_values" in cs and "min" in cs and len(cs["all_values"]) <= 10:
                col_lower = col_name.lower()
                if any(kw in col_lower for kw in ("date", "time", "year", "month")):
                    vals_str = ", ".join(repr(v) for v in cs["all_values"])
                    overview_warnings.append(
                        f"-- ⚠️ {table} ({row_count} rows): {col_name} ONLY contains "
                        f"{vals_str}. Use a DIFFERENT table for other dates!"
                    )

    parts = []
    if overview_warnings:
        parts.append(
            "-- ⚠️ IMPORTANT DATA LIMITATIONS\n"
            + "\n".join(overview_warnings)
            + "\n"
        )

    # Add database-specific domain hints
    db_name = db_path.parent.name if db_path.parent else ""
    domain_hints = _get_domain_hints(db_name, comment_prefix="-- ")
    if domain_hints:
        parts.append(domain_hints)

    for table in tables:
        if table == "sqlite_sequence":
            continue

        columns = get_table_schema(db_path, table)
        fks = get_foreign_keys(db_path, table)
        table_desc = descriptions.get(table, [])
        desc_map = {d["original_column_name"]: d for d in table_desc}

        stats = all_stats_ddl.get(table) or _get_table_stats(db_path, table)
        stats_str = _format_table_stats(table, stats)

        col_defs = []
        for col in columns:
            col_name = col["name"]
            col_type = col["type"] or "TEXT"
            constraints = []
            if col["pk"]:
                constraints.append("PRIMARY KEY")
            if col["notnull"]:
                constraints.append("NOT NULL")
            if col["default"] is not None:
                constraints.append(f"DEFAULT {col['default']}")

            line = f"    `{col_name}` {col_type}"
            if constraints:
                line += " " + " ".join(constraints)

            # Build comment with description + stats
            comment_parts = []
            if include_comments and col_name in desc_map:
                desc = desc_map[col_name].get("column_description", "")
                if desc:
                    comment_parts.append(desc)

            col_stat = stats.get("columns", {}).get(col_name, {})
            if "all_values" in col_stat:
                vals = col_stat["all_values"]
                vals_str = ", ".join(repr(v) for v in vals[:10])
                comment_parts.append(f"values: {vals_str}")
            if "min" in col_stat and "max" in col_stat:
                comment_parts.append(f"range: [{col_stat['min']}..{col_stat['max']}]")

            if comment_parts:
                line += f"  -- {'; '.join(comment_parts)}"

            col_defs.append(line)

        for fk in fks:
            col_defs.append(
                f"    FOREIGN KEY (`{fk['from_column']}`) "
                f"REFERENCES `{fk['to_table']}`(`{fk['to_column']}`)"
            )

        ddl = f"-- {stats_str}\nCREATE TABLE `{table}` (\n" + ",\n".join(col_defs) + "\n);"
        parts.append(ddl)

    return "\n\n".join(parts)
