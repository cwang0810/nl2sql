"""
Quick smoke test: single model, single generation, no fancy pipeline.
Validates API connectivity + SQL execution + evaluation end-to-end.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.env_loader import load_env_file

# ─── Paths ───
PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data" / "mini_dev" / "minidev" / "MINIDEV"
DATA_JSON = DATA_DIR / "mini_dev_sqlite.json"
GOLD_SQL = DATA_DIR / "mini_dev_sqlite_gold.sql"
DB_ROOT = DATA_DIR / "dev_databases"
CONFIG_YAML = PROJECT / "config" / "config.yaml"

SYSTEM_PROMPT = "You are an expert SQLite SQL developer. Output ONLY the SQL query, nothing else."

PROMPT_TEMPLATE = """Given the database schema and question, write the correct SQLite SQL query.

【Database Schema】
{schema}

【External Knowledge】
{evidence}

【Question】
{question}

Output ONLY the SQL query:"""


def load_config():
    import yaml
    import re
    load_env_file(PROJECT / ".env", override=False)

    with open(CONFIG_YAML) as f:
        cfg = yaml.safe_load(f)

    # Support ${ENV_VAR} in YAML config values.
    env_pat = re.compile(r"^\$\{([A-Z0-9_]+)\}$")

    def resolve(val):
        if isinstance(val, str):
            m = env_pat.match(val.strip())
            if m:
                return os.environ.get(m.group(1), "")
        if isinstance(val, dict):
            return {k: resolve(v) for k, v in val.items()}
        if isinstance(val, list):
            return [resolve(v) for v in val]
        return val

    return resolve(cfg)


def get_schema_ddl(db_path: Path) -> str:
    """Get full DDL from sqlite_master."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL;")
        return "\n\n".join(row[0] for row in cur.fetchall())
    finally:
        conn.close()


def execute_sql(db_path: str, sql: str, timeout: int = 30):
    import sqlite3
    from func_timeout import func_timeout, FunctionTimedOut
    def _run():
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchall()
        finally:
            conn.close()
    try:
        return func_timeout(timeout, _run)
    except FunctionTimedOut:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def extract_sql(text: str) -> str:
    import re
    for pattern in [r"```sql\s*(.*?)```", r"```\s*(.*?)```"]:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    m = re.search(r"((?:SELECT|WITH)\s+.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(";")
    return text.strip().rstrip(";")


async def main():
    cfg = load_config()
    ds_cfg = cfg["dashscope"]
    model_name = cfg["models"]["deepseek"]["model_name"]

    api_key = ds_cfg["api_key"]
    api_base = ds_cfg["api_base"]

    print(f"Model:    {model_name}")
    print(f"API Base: {api_base}")
    print(f"Data:     {DATA_JSON}")
    print(f"DB Root:  {DB_ROOT}")
    print()

    import openai
    client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=120)

    # Load data
    with open(DATA_JSON) as f:
        data = json.load(f)

    # Gold SQL
    gold_sqls = []
    with open(GOLD_SQL) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                gold_sqls.append(parts[0])

    # Test on first N questions
    N = 5
    correct = 0
    total = 0

    for i in range(N):
        item = data[i]
        q = item["question"]
        db_id = item["db_id"]
        evidence = item.get("evidence", "")
        gold_sql = gold_sqls[i]
        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

        schema = get_schema_ddl(db_path)

        prompt = PROMPT_TEMPLATE.format(
            schema=schema,
            evidence=evidence or "None",
            question=q,
        )

        print(f"[{i+1}/{N}] {q[:80]}...")
        print(f"       DB: {db_id}, Difficulty: {item.get('difficulty', '?')}")

        start = time.monotonic()
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            pred_sql = extract_sql(resp.choices[0].message.content or "")
            elapsed = (time.monotonic() - start) * 1000
        except Exception as e:
            print(f"       API ERROR: {e}")
            continue

        # Execute both
        pred_result = execute_sql(str(db_path), pred_sql)
        gold_result = execute_sql(str(db_path), gold_sql)

        is_correct = (
            not isinstance(pred_result, str)
            and not isinstance(gold_result, str)
            and set(pred_result) == set(gold_result)
        )

        total += 1
        if is_correct:
            correct += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"       {status} ({elapsed:.0f}ms)")
        if not is_correct:
            print(f"       Pred: {pred_sql[:120]}")
            print(f"       Gold: {gold_sql[:120]}")
        print()

    print(f"{'='*50}")
    print(f"Result: {correct}/{total} ({correct/total*100:.0f}% EX)")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
