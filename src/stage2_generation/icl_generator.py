"""
ICL Generator: multi-temperature, multi-prompt In-Context Learning SQL generation.
Primary model: DeepSeek V3.2
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..models.base import LLMClient
from ..utils.prompt_loader import load_sql_rules
from .generator_base import SQLCandidate, SQLGenerator

logger = logging.getLogger(__name__)


# ─── Load rules from config (single source of truth) ───
_SQL_RULES = load_sql_rules()

# ─── Prompt templates (rules injected at runtime) ───

DIRECT_PROMPT = """You are an expert SQL developer. Given the database schema and a natural language question, generate the correct SQLite SQL query.

【Database Schema】
{schema}

【Relevant Values】
{retrieved_values}

【External Knowledge】
{evidence}

{sql_rules}
{few_shot_section}

【Question】
{question}

Output ONLY the SQL query, nothing else:"""

COT_PROMPT = """You are an expert SQL developer. Given the database schema and a natural language question, generate the correct SQLite SQL query.

【Database Schema】
{schema}

【Relevant Values】
{retrieved_values}

【External Knowledge】
{evidence}

{sql_rules}
{few_shot_section}

【Question】
{question}

Think step by step:
1. Understand the question: What exactly is being asked? What columns should appear in the output?
2. Identify relevant tables and columns: Which tables are needed? How to join them?
3. Build WHERE conditions: What filters are needed?
4. Determine aggregation and ordering: GROUP BY / ORDER BY / LIMIT?
5. Double-check: Does the SELECT clause return ONLY what the question asks for? No extra columns?
6. Write the final SQL query.

After your analysis, output the final SQL query:"""


class ICLGenerator(SQLGenerator):
    """In-Context Learning generator with multi-temperature, multi-prompt strategy."""

    def __init__(
        self,
        llm_client: LLMClient,
        temperatures: list[float] | None = None,
        prompts: list[str] | None = None,
        candidates_per_config: int = 1,
    ):
        super().__init__(llm_client, name="ICLGenerator")
        self.temperatures = temperatures or [0.0, 0.5, 1.0, 1.5]
        self.prompt_names = prompts or ["direct", "cot"]
        self.candidates_per_config = candidates_per_config

        self.prompt_templates = {
            "direct": DIRECT_PROMPT,
            "icl_direct": DIRECT_PROMPT,
            "cot": COT_PROMPT,
            "icl_cot": COT_PROMPT,
        }

    async def generate(
        self,
        question: str,
        schema: str,
        evidence: str = "",
        few_shot_examples: str = "",
        retrieved_values: str = "",
        **kwargs: Any,
    ) -> list[SQLCandidate]:
        """Generate SQL candidates with multiple temperatures and prompts."""
        few_shot_section = ""
        if few_shot_examples:
            few_shot_section = f"【Similar Examples】\n{few_shot_examples}"

        tasks = []
        configs = []

        for temp in self.temperatures:
            for prompt_name in self.prompt_names:
                template = self.prompt_templates.get(prompt_name, DIRECT_PROMPT)
                prompt = template.format(
                    schema=schema,
                    question=question,
                    evidence=evidence or "No additional knowledge provided.",
                    few_shot_section=few_shot_section,
                    retrieved_values=retrieved_values or "No specific values found.",
                    sql_rules=_SQL_RULES,
                )

                tasks.append(
                    self.llm.generate(
                        prompt=prompt,
                        temperature=temp,
                        n=self.candidates_per_config,
                    )
                )
                configs.append({"temp": temp, "prompt": prompt_name})

        # Run all API calls concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        for config, result in zip(configs, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"ICL generation failed (temp={config['temp']}, "
                    f"prompt={config['prompt']}): {result}"
                )
                continue

            for resp in result:
                sql = resp.sql
                if sql:
                    candidates.append(SQLCandidate(
                        sql=sql,
                        generator=self.name,
                        model=self.llm.model_name,
                        temperature=config["temp"],
                        prompt_strategy=config["prompt"],
                        raw_response=resp.content,
                    ))

        logger.info(f"ICL Generator produced {len(candidates)} candidates")
        return candidates
