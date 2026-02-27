"""Embedding via DashScope API (text-embedding-v3).

No local model, no PyTorch — just an API call through the same DashScope endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
import openai

logger = logging.getLogger(__name__)

_client_cache: dict[str, "EmbeddingModel"] = {}

# DashScope embedding 每次最多 6 个文本
_DASHSCOPE_BATCH_LIMIT = 6


class EmbeddingModel:
    """DashScope embedding API wrapper."""

    def __init__(
        self,
        model_name: str = "text-embedding-v3",
        api_key: str | None = None,
        api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=api_base,
        )
        logger.info(f"Using DashScope embedding model: {model_name}")

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = _DASHSCOPE_BATCH_LIMIT,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings via DashScope API.

        Args:
            texts: List of text strings.
            batch_size: API batch size (DashScope limit: 6).
            normalize: Whether to L2-normalize embeddings.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        texts = list(texts)
        if not texts:
            return np.array([])

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_emb = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_emb)

        result = np.array(all_embeddings, dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            result = result / norms

        return result

    def similarity(self, query: str, candidates: Sequence[str]) -> np.ndarray:
        """Compute cosine similarity between a query and candidates.

        Returns:
            1D array of similarity scores.
        """
        q_emb = self.encode([query])
        c_emb = self.encode(candidates)
        return (q_emb @ c_emb.T).flatten()


def get_embedding_model(
    model_name: str = "text-embedding-v3",
    api_key: str | None = None,
    **kwargs,
) -> EmbeddingModel:
    """Get or create a cached embedding model instance."""
    key = f"{model_name}"
    if key not in _client_cache:
        _client_cache[key] = EmbeddingModel(model_name, api_key=api_key, **kwargs)
    return _client_cache[key]
