"""
Run the NL2SQL pipeline on the BIRD Mini-Dev set.
Quick evaluation loop for development iteration.

Usage:
    python scripts/run_mini_dev.py --config config/config.yaml
    python scripts/run_mini_dev.py --db_root data/mini_dev/databases --data_json data/mini_dev/mini_dev_sqlite.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import NL2SQLPipeline, PipelineConfig
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Run NL2SQL on BIRD Mini-Dev")
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML path")
    parser.add_argument("--data_json", default="data/mini_dev/minidev/MINIDEV/mini_dev_sqlite.json")
    parser.add_argument("--db_root", default="data/mini_dev/minidev/MINIDEV/dev_databases")
    parser.add_argument("--output", default="data/results/mini_dev_results.json")
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    args = parser.parse_args()

    setup_logging("INFO")

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig()

    config.db_root = args.db_root

    # Load data
    with open(args.data_json) as f:
        data = json.load(f)

    if args.limit > 0:
        data = data[:args.limit]

    logger.info(f"Loaded {len(data)} questions from {args.data_json}")

    # Initialize pipeline
    pipeline = NL2SQLPipeline(config)

    # Load example index if available
    index_path = Path("data/indices/example_index")
    if index_path.exists():
        pipeline.load_example_index(index_path)

    # Run
    results = await pipeline.process_dataset(
        data=data,
        db_root=args.db_root,
        output_path=args.output,
        max_concurrent=args.max_concurrent,
        resume_from=args.resume_from,
    )

    # Summary
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    by_method = {}
    by_db = {}
    all_times = []
    error_count = 0

    for r in results:
        by_method.setdefault(r.selected_by, 0)
        by_method[r.selected_by] += 1
        by_db.setdefault(r.db_id, {"count": 0, "total_ms": 0})
        by_db[r.db_id]["count"] += 1
        by_db[r.db_id]["total_ms"] += r.processing_time_ms
        all_times.append(r.processing_time_ms)
        if r.selected_by == "error":
            error_count += 1

    avg_time = sum(all_times) / total
    avg_candidates = sum(r.candidates_count for r in results) / total
    sorted_times = sorted(all_times)
    p50 = sorted_times[int(total * 0.5)]
    p95 = sorted_times[min(int(total * 0.95), total - 1)]

    print(f"\n{'='*70}")
    print(f"  BIRD Mini-Dev Results Summary")
    print(f"{'='*70}")
    print(f"  Questions:       {total}  (errors: {error_count})")
    print(f"  Avg candidates:  {avg_candidates:.1f}")
    print(f"  Selection:       {by_method}")
    print(f"  Timing:          avg={avg_time/1000:.1f}s  p50={p50/1000:.1f}s  p95={p95/1000:.1f}s"
          f"  min={sorted_times[0]/1000:.1f}s  max={sorted_times[-1]/1000:.1f}s")
    print(f"\n  Per-Database:")
    for db in sorted(by_db, key=lambda d: by_db[d]["count"], reverse=True):
        info = by_db[db]
        db_avg = info["total_ms"] / info["count"] / 1000
        print(f"    {db:<30} {info['count']:>4} questions  avg {db_avg:.1f}s")
    print(f"\n  Results: {args.output}")
    print(f"  Submit:  {Path(args.output).with_suffix('.txt')}")
    print(f"\n  Evaluate:")
    print(f"    python evaluation/evaluation_ex.py \\")
    print(f"      --predicted_sql_path {Path(args.output).with_suffix('.txt')} \\")
    print(f"      --ground_truth_path data/mini_dev/mini_dev_sqlite_gold.sql \\")
    print(f"      --db_root_path {args.db_root} \\")
    print(f"      --diff_json_path {args.data_json}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
