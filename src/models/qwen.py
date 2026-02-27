"""Qwen3 Coder API client via DashScope (阿里百炼)."""

from __future__ import annotations

import os

from .base import LLMClient

# DashScope 统一入口
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenClient(LLMClient):
    """Qwen3 Coder client via DashScope OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = DASHSCOPE_API_BASE,
        model_name: str = "qwen3-coder-480b-a35b-instruct",
        **kwargs,
    ):
        super().__init__(
            api_base=api_base,
            api_key=api_key or os.environ["DASHSCOPE_API_KEY"],
            model_name=model_name,
            **kwargs,
        )

    def name(self) -> str:
        return f"Qwen3({self.model_name})"
