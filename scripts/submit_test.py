"""
Generate BIRD test set submission file.

Usage:
    python scripts/submit_test.py --config config/config.yaml --data_json data/bird_test/test.json --db_root data/bird_test/test_databases
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import NL2SQLPipeline, PipelineConfig
from src.utils.logger import setup_logging


async def main():
    parser = argparse.ArgumentParser(description="Generate BIRD test submission")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data_json", required=True, help="Test data JSON")
    parser.add_argument("--db_root", required=True, help="Test database root")
    parser.add_argument("--output", default="data/results/test_submission.txt")
    parser.add_argument("--max_concurrent", type=int, default=5)
    args = parser.parse_args()

    setup_logging("INFO")

    config = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    config.db_root = args.db_root

    with open(args.data_json) as f:
        data = json.load(f)

    pipeline = NL2SQLPipeline(config)

    index_path = Path("data/indices/example_index")
    if index_path.exists():
        pipeline.load_example_index(index_path)

    results = await pipeline.process_dataset(
        data=data,
        db_root=args.db_root,
        output_path=args.output.replace(".txt", ".json"),
        max_concurrent=args.max_concurrent,
    )

    # BIRD submission format
    with open(args.output, "w") as f:
        for r in results:
            f.write(f"{r.predicted_sql}\t----- bird -----\t{r.db_id}\n")

    print(f"\nSubmission file generated: {args.output}")
    print(f"Total questions: {len(results)}")
    print(f"\nSubmit to: bird.bench23@gmail.com")


if __name__ == "__main__":
    asyncio.run(main())
