"""
BIRD Execution Accuracy (EX) evaluation.
Adapted from the official BIRD-bench evaluation script.

Usage:
    python evaluation_ex.py \
        --predicted_sql_path results/predict.json \
        --ground_truth_path data/mini_dev/mini_dev_sqlite_gold.sql \
        --db_root_path data/mini_dev/mini_dev_sqlite/databases \
        --diff_json_path data/mini_dev/mini_dev_sqlite.json \
        --num_cpus 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout

logger = logging.getLogger(__name__)


def execute_sql(db_path: str, sql: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        conn.close()


def execute_with_timeout(db_path: str, sql: str, timeout: int = 30):
    try:
        result = func_timeout(timeout, execute_sql, args=(db_path, sql))
        return result
    except FunctionTimedOut:
        return "timeout"
    except Exception as e:
        return f"error: {e}"


def evaluate_one(args: tuple) -> dict:
    """Evaluate a single (predicted, gold) SQL pair."""
    idx, predicted_sql, gold_sql, db_path, timeout = args

    pred_result = execute_with_timeout(db_path, predicted_sql, timeout)
    gold_result = execute_with_timeout(db_path, gold_sql, timeout)

    if isinstance(pred_result, str) or isinstance(gold_result, str):
        return {"idx": idx, "correct": 0, "pred_result": str(pred_result), "gold_result": str(gold_result)}

    correct = 1 if set(pred_result) == set(gold_result) else 0
    return {"idx": idx, "correct": correct, "pred_result": str(pred_result)[:200], "gold_result": str(gold_result)[:200]}


def load_predictions(path: str) -> list[dict]:
    """Load predictions from JSON file.

    Expected format: list of {"sql": "...", "db_id": "..."}
    Or BIRD format: "SQL\t----- bird -----\tdb_id" per line.
    """
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        # BIRD text format
        results = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "----- bird -----" in line:
                    parts = line.split("----- bird -----")
                    results.append({
                        "sql": parts[0].strip(),
                        "db_id": parts[1].strip(),
                    })
                else:
                    parts = line.split("\t")
                    results.append({
                        "sql": parts[0].strip(),
                        "db_id": parts[1].strip() if len(parts) > 1 else "",
                    })
        return results


def load_gold(path: str) -> list[dict]:
    """Load gold SQL from file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            results.append({
                "sql": parts[0].strip(),
                "db_id": parts[1].strip() if len(parts) > 1 else "",
            })
    return results


def compute_accuracy_by_difficulty(
    results: list[dict],
    diff_json_path: str | None = None,
) -> dict:
    """Compute accuracy overall and by difficulty level."""
    total = len(results)
    correct = sum(r["correct"] for r in results)
    overall = correct / total * 100 if total > 0 else 0

    output = {"overall": {"accuracy": overall, "correct": correct, "total": total}}

    if diff_json_path:
        with open(diff_json_path) as f:
            diff_data = json.load(f)

        difficulty_map = {}
        for i, item in enumerate(diff_data):
            difficulty_map[i] = item.get("difficulty", "unknown")

        by_diff = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            diff = difficulty_map.get(r["idx"], "unknown")
            by_diff[diff]["total"] += 1
            by_diff[diff]["correct"] += r["correct"]

        for diff, stats in by_diff.items():
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            output[diff] = {"accuracy": acc, **stats}

    return output


def evaluate(
    predicted_sql_path: str,
    ground_truth_path: str,
    db_root_path: str,
    diff_json_path: str | None = None,
    num_cpus: int = 8,
    timeout: int = 30,
    output_log_path: str | None = None,
) -> dict:
    """Run EX evaluation.

    Returns:
        Dict with accuracy metrics.
    """
    predictions = load_predictions(predicted_sql_path)
    golds = load_gold(ground_truth_path)

    assert len(predictions) == len(golds), (
        f"Prediction count ({len(predictions)}) != Gold count ({len(golds)})"
    )

    eval_args = []
    for i, (pred, gold) in enumerate(zip(predictions, golds)):
        db_id = pred.get("db_id") or gold["db_id"]
        db_path = str(Path(db_root_path) / db_id / f"{db_id}.sqlite")
        eval_args.append((i, pred["sql"], gold["sql"], db_path, timeout))

    with Pool(num_cpus) as pool:
        results = pool.map(evaluate_one, eval_args)

    results.sort(key=lambda x: x["idx"])
    accuracy = compute_accuracy_by_difficulty(results, diff_json_path)

    if output_log_path:
        with open(output_log_path, "w") as f:
            json.dump({"accuracy": accuracy, "details": results}, f, indent=2)

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="BIRD EX Evaluation")
    parser.add_argument("--predicted_sql_path", required=True)
    parser.add_argument("--ground_truth_path", required=True)
    parser.add_argument("--db_root_path", required=True)
    parser.add_argument("--diff_json_path", default=None)
    parser.add_argument("--num_cpus", type=int, default=8)
    parser.add_argument("--meta_time_out", type=int, default=30)
    parser.add_argument("--output_log_path", default=None)
    args = parser.parse_args()

    result = evaluate(
        predicted_sql_path=args.predicted_sql_path,
        ground_truth_path=args.ground_truth_path,
        db_root_path=args.db_root_path,
        diff_json_path=args.diff_json_path,
        num_cpus=args.num_cpus,
        timeout=args.meta_time_out,
        output_log_path=args.output_log_path,
    )

    print("\n=== BIRD EX Evaluation Results ===")
    for key, val in result.items():
        print(f"  {key}: {val['accuracy']:.2f}% ({val['correct']}/{val['total']})")


if __name__ == "__main__":
    main()
