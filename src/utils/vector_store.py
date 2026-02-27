"""Vector store for similarity search (FAISS-based for speed)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with score and metadata."""
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """FAISS-based vector store for fast similarity search."""

    def __init__(self):
        self._index = None
        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    def build(
        self,
        texts: Sequence[str],
        embeddings: np.ndarray,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Build the index from texts and pre-computed embeddings.

        Args:
            texts: Text strings corresponding to each embedding.
            embeddings: numpy array of shape (n, dim).
            metadata: Optional metadata dicts for each text.
        """
        import faiss

        n, dim = embeddings.shape
        self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)
        self._index.add(embeddings.astype(np.float32))
        self._texts = list(texts)
        self._metadata = list(metadata) if metadata else [{} for _ in texts]
        logger.info(f"Built vector store with {n} vectors of dim {dim}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for the top-k most similar vectors.

        Args:
            query_embedding: Query vector of shape (1, dim) or (dim,).
            top_k: Number of results to return.

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        top_k = min(top_k, len(self._texts))
        scores, indices = self._index.search(
            query_embedding.astype(np.float32), top_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(
                text=self._texts[idx],
                score=float(score),
                metadata=self._metadata[idx],
            ))
        return results

    def save(self, path: str | Path) -> None:
        """Save the vector store to disk."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "data.json", "w") as f:
            json.dump(
                {"texts": self._texts, "metadata": self._metadata},
                f,
                ensure_ascii=False,
            )
        logger.info(f"Saved vector store to {path}")

    def load(self, path: str | Path) -> None:
        """Load the vector store from disk."""
        import faiss

        path = Path(path)
        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "data.json") as f:
            data = json.load(f)
        self._texts = data["texts"]
        self._metadata = data["metadata"]
        logger.info(
            f"Loaded vector store from {path} ({len(self._texts)} vectors)"
        )

    @property
    def size(self) -> int:
        return len(self._texts)
