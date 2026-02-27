"""
Cell value retrieval: find database values matching entities in the question.
Solves the "dirty data" problem in BIRD.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..utils.db_executor import get_table_names, get_table_schema
from ..utils.embedding import EmbeddingModel, get_embedding_model
from ..utils.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ValueRetriever:
    """Retrieve relevant database cell values for a question using embedding search."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        top_k: int = 5,
        max_values_per_column: int = 1000,
    ):
        self.emb = embedding_model or get_embedding_model()
        self.top_k = top_k
        self.max_values = max_values_per_column
        self._store_cache: dict[str, VectorStore] = {}

    def _build_value_store(self, db_path: Path) -> VectorStore:
        """Build a vector store of all text values in the database."""
        import sqlite3

        cache_key = str(db_path)
        if cache_key in self._store_cache:
            return self._store_cache[cache_key]

        conn = sqlite3.connect(str(db_path))
        texts = []
        metadata = []

        try:
            tables = get_table_names(db_path)
            for table in tables:
                columns = get_table_schema(db_path, table)
                for col in columns:
                    col_name = col["name"]
                    col_type = (col["type"] or "").upper()

                    # Only index text-like columns
                    if any(t in col_type for t in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "BOOL")):
                        continue

                    try:
                        cursor = conn.cursor()
                        cursor.execute(
                            f"SELECT DISTINCT `{col_name}` FROM `{table}` "
                            f"WHERE `{col_name}` IS NOT NULL "
                            f"AND typeof(`{col_name}`) = 'text' "
                            f"LIMIT ?",
                            (self.max_values,),
                        )
                        for (val,) in cursor.fetchall():
                            val_str = str(val).strip()
                            if val_str and len(val_str) < 200:
                                texts.append(val_str)
                                metadata.append({
                                    "table": table,
                                    "column": col_name,
                                    "value": val_str,
                                })
                    except Exception:
                        continue
        finally:
            conn.close()

        store = VectorStore()
        if texts:
            embeddings = self.emb.encode(texts)
            store.build(texts, embeddings, metadata)
            logger.info(f"Built value store for {db_path.name}: {len(texts)} values")
        else:
            logger.warning(f"No text values found in {db_path.name}")

        self._store_cache[cache_key] = store
        return store

    def retrieve(
        self,
        question: str,
        db_path: str | Path,
        top_k: int | None = None,
    ) -> list[dict[str, str]]:
        """Retrieve database values matching the question.

        Args:
            question: Natural language question.
            db_path: Path to SQLite database.
            top_k: Number of results to return.

        Returns:
            List of dicts with "table", "column", "value" keys.
        """
        db_path = Path(db_path)
        top_k = top_k or self.top_k
        store = self._build_value_store(db_path)

        if store.size == 0:
            return []

        query_emb = self.emb.encode([question])
        results = store.search(query_emb, top_k=top_k)

        return [
            {
                "table": r.metadata["table"],
                "column": r.metadata["column"],
                "value": r.metadata["value"],
                "score": r.score,
            }
            for r in results
        ]

    def format_values(self, values: list[dict[str, str]]) -> str:
        """Format retrieved values as a string for prompt injection."""
        if not values:
            return "No specific values found."

        lines = []
        for v in values:
            lines.append(f"- {v['table']}.{v['column']} contains value: \"{v['value']}\"")
        return "\n".join(lines)
