"""
Schema Linking: identify relevant tables and columns for a question.

Two-stage approach:
1. Embedding-based retrieval: fast initial filtering
2. LLM-based refinement: precise selection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..models.base import LLMClient
from ..utils.db_executor import get_table_names, get_table_schema, get_foreign_keys
from ..utils.embedding import EmbeddingModel, get_embedding_model
from ..utils.vector_store import VectorStore

logger = logging.getLogger(__name__)

SCHEMA_LINK_PROMPT = """You are a database expert. Given a natural language question and a database schema, identify which tables and columns are needed to answer the question.

【Database Schema Overview】
{schema_overview}

【Question】
{question}

【External Knowledge】
{evidence}

【Candidate Columns (from similarity search)】
{candidate_columns}

Please output the relevant tables and columns in the following JSON format:
```json
{{
  "tables": ["table1", "table2"],
  "columns": [
    {{"table": "table1", "column": "col1"}},
    {{"table": "table1", "column": "col2"}},
    {{"table": "table2", "column": "col3"}}
  ]
}}
```

Only include tables and columns that are truly needed. Include join columns (foreign keys) if multiple tables are needed."""


class SchemaLinker:
    """Two-stage schema linker: embedding retrieval + LLM refinement."""

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_model: EmbeddingModel | None = None,
        embedding_top_k: int = 30,
    ):
        self.llm = llm_client
        self.emb = embedding_model or get_embedding_model()
        self.top_k = embedding_top_k

    def _build_column_descriptions(
        self,
        db_path: Path,
        db_dir: Path | None = None,
    ) -> list[dict[str, str]]:
        """Build a flat list of column descriptions for embedding search."""
        from .schema_formatter import load_database_descriptions

        tables = get_table_names(db_path)
        descriptions = load_database_descriptions(db_dir) if db_dir else {}

        columns = []
        for table in tables:
            schema = get_table_schema(db_path, table)
            desc_map = {}
            if table in descriptions:
                desc_map = {d["original_column_name"]: d for d in descriptions[table]}

            for col in schema:
                col_name = col["name"]
                desc = desc_map.get(col_name, {})
                text = (
                    f"Table: {table}, Column: {col_name}, "
                    f"Type: {col['type']}, "
                    f"Description: {desc.get('column_description', '')}, "
                    f"Values: {desc.get('value_description', '')}"
                )
                columns.append({
                    "table": table,
                    "column": col_name,
                    "type": col["type"],
                    "text": text,
                })
        return columns

    def _build_schema_overview(self, db_path: Path) -> str:
        """Build a concise schema overview for LLM context."""
        tables = get_table_names(db_path)
        lines = []
        for table in tables:
            cols = get_table_schema(db_path, table)
            col_names = [c["name"] for c in cols]
            lines.append(f"- {table}: {', '.join(col_names)}")
        return "\n".join(lines)

    async def link(
        self,
        question: str,
        db_path: str | Path,
        db_dir: Path | None = None,
        evidence: str = "",
    ) -> dict[str, Any]:
        """Identify relevant tables and columns for a question.

        Args:
            question: Natural language question.
            db_path: Path to SQLite database.
            db_dir: Path to BIRD database directory (with database_description/).
            evidence: External knowledge / evidence string.

        Returns:
            Dict with "tables" and "columns" lists.
        """
        db_path = Path(db_path)

        # Stage 1: Embedding retrieval
        all_columns = self._build_column_descriptions(db_path, db_dir)
        if not all_columns:
            return {"tables": [], "columns": []}

        texts = [c["text"] for c in all_columns]
        scores = self.emb.similarity(question, texts)

        top_indices = scores.argsort()[::-1][:self.top_k]
        candidates = [all_columns[i] for i in top_indices]

        candidate_text = "\n".join(
            f"- {c['table']}.{c['column']} ({c['type']}): {c['text']}"
            for c in candidates
        )

        # Stage 2: LLM refinement
        schema_overview = self._build_schema_overview(db_path)
        prompt = SCHEMA_LINK_PROMPT.format(
            schema_overview=schema_overview,
            question=question,
            evidence=evidence,
            candidate_columns=candidate_text,
        )

        try:
            responses = await self.llm.generate(prompt, temperature=0.0)
            import json
            import re

            content = responses[0].content
            # Extract JSON from response
            match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
            if match:
                result = json.loads(match.group(1))
            else:
                result = json.loads(content)

            return {
                "tables": result.get("tables", []),
                "columns": result.get("columns", []),
            }
        except Exception as e:
            logger.warning(f"LLM schema linking failed: {e}, using embedding results")
            # Fallback to embedding-only results
            tables = list({c["table"] for c in candidates[:15]})
            columns = [{"table": c["table"], "column": c["column"]} for c in candidates[:15]]
            return {"tables": tables, "columns": columns}
