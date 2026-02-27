"""
Build vector indices for schema, values, and few-shot examples.
Run this after downloading the BIRD dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage1_understanding.example_retriever import ExampleRetriever
from src.utils.embedding import get_embedding_model
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def build_example_index(train_json: str, output_dir: str) -> None:
    """Build the few-shot example retrieval index from training data."""
    output_path = Path(output_dir) / "example_index"

    logger.info(f"Building example index from {train_json}...")
    retriever = ExampleRetriever()
    retriever.build_index(train_json)
    retriever.save_index(output_path)
    logger.info(f"Example index saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build vector indices for BIRD NL2SQL")
    parser.add_argument(
        "--train_json",
        default="data/bird_train/train.json",
        help="Path to BIRD training data JSON",
    )
    parser.add_argument(
        "--output_dir",
        default="data/indices",
        help="Directory to save indices",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if Path(args.train_json).exists():
        build_example_index(args.train_json, args.output_dir)
    else:
        logger.warning(f"Training data not found at {args.train_json}, skipping example index")
        logger.info("Download training data first: bash scripts/setup_data.sh")


if __name__ == "__main__":
    main()
