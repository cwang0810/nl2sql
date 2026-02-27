"""
Error analysis tool: analyze prediction errors on BIRD dev/mini-dev.

Categorizes errors by:
- Difficulty level
- Error type (empty result, wrong result, timeout, execution error)
- Generator source
- Database domain
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def execute_sql(db_path: str, sql: str, timeout: int = 30):
    def _exec():
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchall()
        finally:
            conn.close()

    try:
        return func_timeout(timeout, _exec)
    except FunctionTimedOut:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def categorize_error(pred_result, gold_result) -> str:
    if isinstance(pred_result, str):
        if pred_result == "TIMEOUT":
            return "timeout"
        return "execution_error"
    if isinstance(gold_result, str):
        return "gold_error"
    if not pred_result:
        return "empty_result"
    if set(pred_result) != set(gold_result):
        return "wrong_result"
    return "correct"


def main():
    parser = argparse.ArgumentParser(description="Analyze BIRD prediction errors")
    parser.add_argument("--predictions", required=True, help="Predictions JSON file")
    parser.add_argument("--gold", required=True, help="Gold SQL file")
    parser.add_argument("--data_json", required=True, help="BIRD data JSON (with difficulty)")
    parser.add_argument("--db_root", required=True, help="Database root path")
    parser.add_argument("--output", default=None, help="Output analysis JSON")
    args = parser.parse_args()

    setup_logging("INFO")

    # Load data
    with open(args.predictions) as f:
        preds = json.load(f)
    with open(args.data_json) as f:
        data = json.load(f)

    gold_sqls = []
    with open(args.gold) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                gold_sqls.append({"sql": parts[0], "db_id": parts[1] if len(parts) > 1 else ""})

    assert len(preds) == len(gold_sqls) == len(data), "Data length mismatch"

    # Analyze
    error_types = Counter()
    by_difficulty = defaultdict(Counter)
    by_db = defaultdict(Counter)
    errors_detail = []

    for i, (pred, gold, item) in enumerate(zip(preds, gold_sqls, data)):
        db_id = pred.get("db_id", gold["db_id"])
        db_path = str(Path(args.db_root) / db_id / f"{db_id}.sqlite")
        difficulty = item.get("difficulty", "unknown")

        pred_sql = pred.get("predicted_sql", pred.get("sql", ""))
        gold_sql = gold["sql"]

        pred_result = execute_sql(db_path, pred_sql)
        gold_result = execute_sql(db_path, gold_sql)

        category = categorize_error(pred_result, gold_result)
        error_types[category] += 1
        by_difficulty[difficulty][category] += 1
        by_db[db_id][category] += 1

        if category != "correct":
            errors_detail.append({
                "idx": i,
                "question": item["question"],
                "db_id": db_id,
                "difficulty": difficulty,
                "error_type": category,
                "pred_sql": pred_sql,
                "gold_sql": gold_sql,
                "evidence": item.get("evidence", ""),
            })

    # Print summary
    total = len(preds)
    correct = error_types.get("correct", 0)

    print(f"\n{'='*60}")
    print(f"Error Analysis Summary")
    print(f"{'='*60}")
    print(f"Total: {total}, Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"\nError distribution:")
    for err_type, count in error_types.most_common():
        print(f"  {err_type}: {count} ({count/total*100:.1f}%)")

    print(f"\nBy difficulty:")
    for diff in ["simple", "moderate", "challenging"]:
        if diff in by_difficulty:
            stats = by_difficulty[diff]
            diff_total = sum(stats.values())
            diff_correct = stats.get("correct", 0)
            print(f"  {diff}: {diff_correct}/{diff_total} ({diff_correct/diff_total*100:.1f}%)")

    # Save detailed analysis
    if args.output:
        analysis = {
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": correct / total * 100,
                "error_types": dict(error_types),
            },
            "by_difficulty": {k: dict(v) for k, v in by_difficulty.items()},
            "errors": errors_detail[:100],  # Top 100 errors
        }
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed analysis saved to {args.output}")


if __name__ == "__main__":
    main()
