"""Few-shot example retrieval from BIRD training set."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..utils.embedding import EmbeddingModel, get_embedding_model
from ..utils.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ExampleRetriever:
    """Retrieve similar training examples as few-shot demonstrations."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        top_k: int = 3,
    ):
        self.emb = embedding_model or get_embedding_model()
        self.top_k = top_k
        self._store: VectorStore | None = None
        self._examples: list[dict] = []

    def build_index(self, train_json_path: str | Path) -> None:
        """Build vector index from BIRD training data.

        Args:
            train_json_path: Path to train.json (BIRD training set).
        """
        with open(train_json_path) as f:
            data = json.load(f)

        self._examples = data
        questions = [item["question"] for item in data]
        embeddings = self.emb.encode(questions)

        metadata = [
            {
                "idx": i,
                "question": item["question"],
                "sql": item.get("SQL", ""),
                "evidence": item.get("evidence", ""),
                "db_id": item.get("db_id", ""),
            }
            for i, item in enumerate(data)
        ]

        self._store = VectorStore()
        self._store.build(questions, embeddings, metadata)
        logger.info(f"Built example index with {len(data)} training examples")

    def save_index(self, path: str | Path) -> None:
        """Save the index to disk for faster loading."""
        if self._store:
            self._store.save(path)

    def load_index(self, path: str | Path) -> None:
        """Load a pre-built index from disk."""
        self._store = VectorStore()
        self._store.load(path)

    def retrieve(
        self,
        question: str,
        db_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve similar training examples.

        Args:
            question: Test question.
            db_id: Optional database ID to prefer same-domain examples.
            top_k: Number of examples to return.

        Returns:
            List of example dicts with question, sql, evidence, db_id.
        """
        if self._store is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        top_k = top_k or self.top_k
        query_emb = self.emb.encode([question])

        # Retrieve more candidates, then optionally prefer same-domain
        candidates = self._store.search(query_emb, top_k=top_k * 3)

        results = []
        for r in candidates:
            results.append({
                "question": r.metadata["question"],
                "sql": r.metadata["sql"],
                "evidence": r.metadata["evidence"],
                "db_id": r.metadata["db_id"],
                "score": r.score,
            })

        # Prefer same-domain examples, but don't require it
        if db_id:
            same_domain = [r for r in results if r["db_id"] == db_id]
            other_domain = [r for r in results if r["db_id"] != db_id]
            results = same_domain + other_domain

        return results[:top_k]

    def format_examples(self, examples: list[dict[str, Any]]) -> str:
        """Format retrieved examples as few-shot demonstrations."""
        if not examples:
            return ""

        parts = []
        for i, ex in enumerate(examples, 1):
            part = f"Example {i}:\n"
            part += f"Question: {ex['question']}\n"
            if ex.get("evidence"):
                part += f"Knowledge: {ex['evidence']}\n"
            part += f"SQL: {ex['sql']}\n"
            parts.append(part)

        return "\n".join(parts)
