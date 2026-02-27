"""Logging configuration with rich formatting."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging with rich formatting.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.
    """
    handlers: list[logging.Handler] = []

    try:
        from rich.logging import RichHandler
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
    except ImportError:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Lower pipeline sub-component loggers to WARNING during dataset runs
    # (progress is now reported via print statements in process_dataset)
    for name in (
        "src.stage2_generation.icl_generator",
        "src.stage2_generation.cot_generator",
        "src.stage2_generation.sql_fixer",
        "src.stage2_generation.sql_revisor",
        "src.stage3_selection.deduplicator",
        "src.stage3_selection.self_consistency",
        "src.stage3_selection.tournament_selector",
        "src.stage1_understanding.value_retriever",
        "src.utils.vector_store",
        "faiss.loader",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
