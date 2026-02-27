"""
CoT Generator: Chain-of-Thought reasoning SQL generation.
Primary model: Qwen3 480B Coder

Implements multiple reasoning strategies:
- Query Execution Plan CoT (from CHASE-SQL)
- Step-by-step Reasoning
- Divide & Conquer
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..models.base import LLMClient
from ..utils.prompt_loader import load_sql_rules
from .generator_base import SQLCandidate, SQLGenerator

logger = logging.getLogger(__name__)


# ─── Load rules from config (single source of truth) ───
_SQL_RULES = load_sql_rules()

# ─── Prompt templates (rules injected at runtime via {sql_rules}) ───

EXECUTION_PLAN_PROMPT = """You are an expert SQL developer and database engineer. Given the database schema and a natural language question, generate the correct SQLite SQL query by simulating a query execution plan.

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

Simulate the database engine's query execution plan:
1. FROM/JOIN: Which tables to scan and how to join them?
2. WHERE: What filter conditions to apply?
3. GROUP BY: Need any grouping?
4. HAVING: Any post-grouping filters?
5. SELECT: Which columns or aggregate functions? (Return ONLY what is asked!)
6. ORDER BY / LIMIT: Final sorting and limiting?

Following this execution plan, generate the SQL query:"""


STEP_BY_STEP_PROMPT = """You are an expert SQL developer. Given the database schema and a natural language question, generate the correct SQLite SQL query.

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

Let's solve this step by step:
1. Identify the target: What does the question ask for? (column/aggregation)
2. Identify the source: Which tables contain this information?
3. Identify conditions: What filtering conditions are mentioned?
4. Identify relationships: How do the tables connect? (JOIN conditions)
5. Identify extras: Any GROUP BY, ORDER BY, LIMIT, HAVING, subqueries needed?
6. Compose the final SQL.

After your reasoning, output the final SQL query:"""


DIVIDE_CONQUER_PROMPT = """You are an expert SQL developer. Given the database schema and a natural language question, solve it by breaking it into sub-problems.

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

This might be complex. Break it into sub-problems:
1. Identify each independent information need in the question.
2. Write a sub-query or CTE for each sub-problem.
3. Combine sub-queries into the final query.

Solve step by step, then output the final SQL query:"""


DIRECT_PROMPT = """You are an expert SQL developer. Generate the correct SQLite SQL query.

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

Output ONLY the SQL query:"""


class CoTGenerator(SQLGenerator):
    """Chain-of-Thought generator with multiple reasoning strategies."""

    def __init__(
        self,
        llm_client: LLMClient,
        temperatures: list[float] | None = None,
        prompts: list[str] | None = None,
        candidates_per_config: int = 1,
    ):
        super().__init__(llm_client, name="CoTGenerator")
        self.temperatures = temperatures or [0.0, 0.5, 1.0, 1.5]
        self.prompt_names = prompts or ["execution_plan", "divide_conquer"]
        self.candidates_per_config = candidates_per_config

        self.prompt_templates = {
            "execution_plan": EXECUTION_PLAN_PROMPT,
            "cot_execution_plan": EXECUTION_PLAN_PROMPT,
            "step_by_step": STEP_BY_STEP_PROMPT,
            "divide_conquer": DIVIDE_CONQUER_PROMPT,
            "direct": DIRECT_PROMPT,
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
        """Generate SQL candidates with multiple CoT strategies."""
        few_shot_section = ""
        if few_shot_examples:
            few_shot_section = f"【Similar Examples】\n{few_shot_examples}"

        tasks = []
        configs = []

        for temp in self.temperatures:
            for prompt_name in self.prompt_names:
                template = self.prompt_templates.get(prompt_name, EXECUTION_PLAN_PROMPT)
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

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        for config, result in zip(configs, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"CoT generation failed (temp={config['temp']}, "
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

        logger.info(f"CoT Generator produced {len(candidates)} candidates")
        return candidates
