"""
Prompt configuration loader.

Three-tier rule architecture:
  Tier 1 (universal)  — applies to ANY NL2SQL task
  Tier 2 (benchmark)  — specific to BIRD / Spider / WikiSQL conventions
  Tier 3 (database)   — specific to a single database schema (domain_hints.yaml)

Single source of truth — no hardcoded rules in generator code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Cached config directory
_CONFIG_DIR: Path | None = None


def _find_config_dir() -> Path:
    """Find the config directory by walking up from this file."""
    global _CONFIG_DIR
    if _CONFIG_DIR is not None:
        return _CONFIG_DIR
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / "config"
        if candidate.is_dir():
            _CONFIG_DIR = candidate
            return candidate
        current = current.parent
    raise FileNotFoundError("Cannot find config/ directory")


def load_sql_rules(benchmark: str | None = None, config_dir: Path | None = None) -> str:
    """Load SQL rules from config/sql_rules.yaml and format as prompt text.

    Assembles: universal rules + active benchmark rules.

    Args:
        benchmark: Override benchmark name (default: read from yaml 'active_benchmark').
        config_dir: Override config directory.

    Returns:
        Formatted rules string ready to inject into prompts.
    """
    config_dir = config_dir or _find_config_dir()
    rules_path = config_dir / "sql_rules.yaml"

    if not rules_path.exists():
        logger.warning(f"sql_rules.yaml not found at {rules_path}")
        return ""

    with open(rules_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Tier 1: Universal rules
    universal = data.get("universal", [])

    # Tier 2: Benchmark-specific rules
    benchmark = benchmark or data.get("active_benchmark", "bird")
    bench_config = data.get("benchmarks", {}).get(benchmark, {})

    bench_rules = []
    for key in ("dialect_rules", "evidence_rules", "evaluation_rules"):
        bench_rules.extend(bench_config.get(key, []))

    # Assemble into formatted string
    all_rules = universal + bench_rules
    if not all_rules:
        return ""

    lines = ["【CRITICAL RULES — you MUST follow ALL of these】"]
    for i, rule in enumerate(all_rules, 1):
        content = rule.get("content", "").strip()
        lines.append(f"{i}. {content}")

    return "\n".join(lines) + "\n"


def load_domain_hints(db_name: str, config_dir: Path | None = None, comment_prefix: str = "") -> str:
    """Load database-specific domain hints from config/domain_hints.yaml (Tier 3).

    Args:
        db_name: Database identifier (e.g., 'thrombosis_prediction').
        config_dir: Override config directory.
        comment_prefix: Prefix for each line (e.g., '-- ' for SQL comments).

    Returns:
        Formatted domain hints string, or empty string if no hints for this database.
    """
    config_dir = config_dir or _find_config_dir()
    hints_path = config_dir / "domain_hints.yaml"

    if not hints_path.exists():
        return ""

    with open(hints_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    databases = data.get("databases", {})
    db_hints = databases.get(db_name, [])
    if not db_hints:
        return ""

    prefix = comment_prefix or ""
    lines = [f"{prefix}# DATABASE-SPECIFIC DOMAIN KNOWLEDGE"]
    for h in db_hints:
        lines.append(f"{prefix}{h.strip()}")
    return "\n".join(lines) + "\n"
