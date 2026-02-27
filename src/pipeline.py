"""
Main NL2SQL Pipeline: orchestrates the three-stage agentic pipeline.

Stage 1: Task Understanding (Schema Linking + Value Retrieval + Evidence + Few-shot)
Stage 2: Multi-path SQL Candidate Generation + Iterative Refinement
Stage 3: Intelligent SQL Selection (Deduplication + Tournament + Self-consistency)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models.base import LLMClient
from .models.deepseek import DeepSeekClient
from .models.qwen import QwenClient
from .stage1_understanding.evidence_integrator import EvidenceIntegrator
from .stage1_understanding.example_retriever import ExampleRetriever
from .stage1_understanding.schema_formatter import format_ddl_schema, format_light_schema
from .stage1_understanding.schema_linker import SchemaLinker
from .stage1_understanding.value_retriever import ValueRetriever
from .utils.embedding import get_embedding_model
from .stage2_generation.cot_generator import CoTGenerator
from .stage2_generation.generator_base import SQLCandidate
from .stage2_generation.icl_generator import ICLGenerator
from .stage2_generation.sql_fixer import SQLFixer
from .stage2_generation.sql_revisor import SQLRevisor
from .stage3_selection.deduplicator import Deduplicator
from .stage3_selection.self_consistency import SelfConsistencySelector
from .stage3_selection.tournament_selector import TournamentSelector
from .utils.db_executor import ExecutionResult, execute_sql, get_db_path
from .utils.env_loader import load_env_file

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the NL2SQL pipeline."""
    # DashScope API (阿里百炼，两个模型共享同一入口)
    dashscope_api_key: str = ""
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    deepseek_model: str = "deepseek-v3.2"
    qwen_model: str = "qwen3-coder-480b-a35b-instruct"

    # Stage 2 configs
    icl_temperatures: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5])
    cot_temperatures: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5])
    icl_prompts: list[str] = field(default_factory=lambda: ["direct", "cot"])
    cot_prompts: list[str] = field(default_factory=lambda: ["execution_plan", "divide_conquer"])

    # Stage 3 configs
    use_tournament: bool = True
    use_self_consistency_fallback: bool = True

    # Execution
    sql_timeout: int = 30
    fixer_max_retries: int = 3
    revisor_max_retries: int = 2

    # Data paths
    db_root: str = ""
    train_json: str = ""
    example_index_path: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        import os
        import yaml
        config_path = Path(path).resolve()
        # Load project .env first, then let shell-exported vars override.
        load_env_file(config_path.parent.parent / ".env", override=False)

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        def resolve_env(val: str) -> str:
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                env_var = val[2:-1]
                return os.environ.get(env_var, "")
            return val

        dashscope = raw.get("dashscope", {})
        models = raw.get("models", {})
        ds = models.get("deepseek", {})
        qw = models.get("qwen3", {})
        data = raw.get("data", {})

        return cls(
            dashscope_api_key=resolve_env(dashscope.get("api_key", "")),
            dashscope_api_base=dashscope.get("api_base", cls.dashscope_api_base),
            deepseek_model=ds.get("model_name", cls.deepseek_model),
            qwen_model=qw.get("model_name", cls.qwen_model),
            db_root=data.get("bird_dev", ""),
            train_json=data.get("bird_train", ""),
        )


@dataclass
class PipelineResult:
    """Result from processing a single question."""
    question: str
    db_id: str
    predicted_sql: str
    candidates_count: int
    unique_results_count: int
    selected_by: str
    processing_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class NL2SQLPipeline:
    """Main three-stage NL2SQL pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize LLM clients (both via DashScope)
        self.deepseek = DeepSeekClient(
            api_key=config.dashscope_api_key,
            api_base=config.dashscope_api_base,
            model_name=config.deepseek_model,
        )
        self.qwen = QwenClient(
            api_key=config.dashscope_api_key,
            api_base=config.dashscope_api_base,
            model_name=config.qwen_model,
        )

        # Stage 1 components (pass api_key to embedding-based modules)
        emb_model = get_embedding_model(api_key=config.dashscope_api_key)
        self.schema_linker = SchemaLinker(self.deepseek, embedding_model=emb_model)
        self.value_retriever = ValueRetriever(embedding_model=emb_model)
        self.evidence_integrator = EvidenceIntegrator()
        self.example_retriever = ExampleRetriever(embedding_model=emb_model)

        # Stage 2 components
        self.icl_generator = ICLGenerator(
            self.deepseek,
            temperatures=config.icl_temperatures,
            prompts=config.icl_prompts,
        )
        self.cot_generator = CoTGenerator(
            self.qwen,
            temperatures=config.cot_temperatures,
            prompts=config.cot_prompts,
        )
        self.sql_fixer = SQLFixer(self.deepseek, max_retries=config.fixer_max_retries)
        self.sql_revisor = SQLRevisor(self.qwen, max_retries=config.revisor_max_retries)

        # Stage 3 components
        self.deduplicator = Deduplicator(timeout=config.sql_timeout)
        self.tournament_selector = TournamentSelector(self.qwen)
        self.self_consistency = SelfConsistencySelector()

    def load_example_index(self, path: str | Path | None = None) -> None:
        """Load the pre-built few-shot example index."""
        path = path or self.config.example_index_path
        if path and Path(path).exists():
            self.example_retriever.load_index(path)
            logger.info(f"Loaded example index from {path}")

    async def process_question(
        self,
        question: str,
        db_id: str,
        evidence: str = "",
        db_root: str | None = None,
    ) -> PipelineResult:
        """Process a single question through the full pipeline.

        Args:
            question: Natural language question.
            db_id: Database identifier.
            evidence: External knowledge string.
            db_root: Root directory containing databases.

        Returns:
            PipelineResult with the predicted SQL.
        """
        start = time.monotonic()
        db_root = db_root or self.config.db_root
        db_path = get_db_path(db_root, db_id)
        db_dir = Path(db_root) / db_id

        # ═══ Stage 1: Task Understanding ═══
        logger.debug(f"[Stage 1] Processing: {question[:80]}...")

        # Schema linking (graceful degradation if embedding unavailable)
        relevant_tables = None
        value_text = ""
        few_shot_text = ""
        try:
            values = self.value_retriever.retrieve(question, db_path)
            value_text = self.value_retriever.format_values(values)
        except Exception as e:
            logger.debug(f"Value retrieval skipped: {e}")
        try:
            linked = await self.schema_linker.link(question, db_path, db_dir, evidence)
            relevant_tables = linked.get("tables", None) or None
        except Exception as e:
            logger.debug(f"Schema linking skipped: {e}")

        # Format schemas
        light_schema = format_light_schema(db_path, db_dir, relevant_tables)
        ddl_schema = format_ddl_schema(db_path, db_dir, relevant_tables)

        # Evidence
        evidence_text = self.evidence_integrator.format_evidence(evidence)

        # Few-shot examples
        try:
            examples = self.example_retriever.retrieve(question, db_id)
            few_shot_text = self.example_retriever.format_examples(examples)
        except Exception:
            pass  # Index not loaded

        # ═══ Stage 2: SQL Candidate Generation ═══
        logger.debug(f"[Stage 2] Generating candidates...")

        # Run both generators concurrently
        icl_task = self.icl_generator.generate(
            question=question,
            schema=light_schema,
            evidence=evidence_text,
            few_shot_examples=few_shot_text,
            retrieved_values=value_text,
        )
        cot_task = self.cot_generator.generate(
            question=question,
            schema=ddl_schema,
            evidence=evidence_text,
            few_shot_examples=few_shot_text,
            retrieved_values=value_text,
        )

        icl_candidates, cot_candidates = await asyncio.gather(icl_task, cot_task)
        all_candidates = icl_candidates + cot_candidates

        logger.debug(
            f"Generated {len(all_candidates)} candidates "
            f"(ICL: {len(icl_candidates)}, CoT: {len(cot_candidates)})"
        )

        if not all_candidates:
            elapsed = (time.monotonic() - start) * 1000
            return PipelineResult(
                question=question,
                db_id=db_id,
                predicted_sql="SELECT 1",
                candidates_count=0,
                unique_results_count=0,
                selected_by="fallback",
                processing_time_ms=elapsed,
            )

        # Iterative refinement: fix errors, revise suspicious results
        refined_candidates = []
        for candidate in all_candidates:
            result = execute_sql(candidate.sql, db_path, db_id, self.config.sql_timeout)

            if result.is_error:
                fixed_sql, was_fixed = await self.sql_fixer.fix(
                    sql=candidate.sql,
                    db_path=db_path,
                    db_id=db_id,
                    question=question,
                    schema=ddl_schema,
                )
                if was_fixed:
                    candidate = SQLCandidate(
                        sql=fixed_sql,
                        generator=candidate.generator + "+fixer",
                        model=candidate.model,
                        temperature=candidate.temperature,
                        prompt_strategy=candidate.prompt_strategy,
                    )

            elif self.sql_revisor.should_revise(result):
                revised_sql, was_revised = await self.sql_revisor.revise(
                    sql=candidate.sql,
                    result=result,
                    question=question,
                    schema=ddl_schema,
                    evidence=evidence_text,
                    db_path=db_path,
                    db_id=db_id,
                )
                if was_revised:
                    candidate = SQLCandidate(
                        sql=revised_sql,
                        generator=candidate.generator + "+revisor",
                        model=candidate.model,
                        temperature=candidate.temperature,
                        prompt_strategy=candidate.prompt_strategy,
                    )

            refined_candidates.append(candidate)

        # ═══ Stage 3: SQL Selection ═══
        logger.debug(f"[Stage 3] Selecting from {len(refined_candidates)} candidates...")

        # Deduplicate by execution results
        deduped = self.deduplicator.deduplicate(refined_candidates, db_path, db_id)

        if len(deduped) == 1:
            selected = deduped[0][0]
            selected_by = "single"
        else:
            # Hybrid selection: majority vote if dominant group exists, else tournament
            total_candidates = len(refined_candidates)
            top_group_size = deduped[0][0].metadata.get("group_size", 1)
            majority_ratio = top_group_size / total_candidates if total_candidates > 0 else 0

            if majority_ratio >= 0.5:
                # Strong majority — use self-consistency (deterministic)
                selected = self.self_consistency.select(deduped)
                selected_by = "self_consistency"
            elif self.config.use_tournament and len(deduped) <= 10:
                selected = await self.tournament_selector.select(
                    deduped, question, ddl_schema, evidence_text
                )
                selected_by = "tournament"
            else:
                selected = self.self_consistency.select(deduped)
                selected_by = "self_consistency"

        elapsed = (time.monotonic() - start) * 1000

        logger.debug(
            f"Selected SQL (by {selected_by}): {selected.sql[:100]}... "
            f"[{elapsed:.0f}ms]"
        )

        return PipelineResult(
            question=question,
            db_id=db_id,
            predicted_sql=selected.sql,
            candidates_count=len(all_candidates),
            unique_results_count=len(deduped),
            selected_by=selected_by,
            processing_time_ms=elapsed,
            metadata={
                "generator": selected.generator,
                "model": selected.model,
                "temperature": selected.temperature,
                "prompt_strategy": selected.prompt_strategy,
            },
        )

    async def process_dataset(
        self,
        data: list[dict],
        db_root: str,
        output_path: str | Path,
        max_concurrent: int = 5,
        resume_from: int = 0,
        progress_interval: int = 10,
    ) -> list[PipelineResult]:
        """Process a full dataset with real-time progress reporting.

        Args:
            data: List of BIRD data items.
            db_root: Root directory with databases.
            output_path: Path to save results.
            max_concurrent: Max concurrent question processing.
            resume_from: Index to resume from (for checkpoint recovery).
            progress_interval: Print summary stats every N completed questions.

        Returns:
            List of PipelineResult objects.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results for resume
        results: list[PipelineResult] = []
        if resume_from > 0 and output_path.exists():
            with open(output_path) as f:
                results = json.load(f)
            logger.info(f"Resuming from index {resume_from}, loaded {len(results)} existing results")

        total = len(data)
        to_process = total - resume_from
        semaphore = asyncio.Semaphore(max_concurrent)

        # Progress tracking state
        completed = 0
        start_time = time.monotonic()
        db_counts: dict[str, int] = {}
        method_counts: dict[str, int] = {}
        times: list[float] = []
        lock = asyncio.Lock()

        def _format_time(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.0f}s"
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

        async def _on_complete(idx: int, item: dict, result: PipelineResult) -> None:
            nonlocal completed
            async with lock:
                completed += 1

                # Track stats
                db = item["db_id"]
                db_counts[db] = db_counts.get(db, 0) + 1
                method_counts[result.selected_by] = method_counts.get(result.selected_by, 0) + 1
                times.append(result.processing_time_ms)

                # Per-question line
                elapsed_s = result.processing_time_ms / 1000
                status = "ok" if result.selected_by != "error" else "ERR"
                db_short = db[:20].ljust(20)
                print(
                    f"[{completed:>4}/{to_process}] {status:>3} {db_short} "
                    f"| {result.candidates_count:>2} cands, {result.unique_results_count:>2} unique "
                    f"| {result.selected_by:<18} | {elapsed_s:.1f}s",
                    flush=True,
                )

                # Periodic summary
                if completed % progress_interval == 0 or completed == to_process:
                    wall = time.monotonic() - start_time
                    eta = (wall / completed) * (to_process - completed) if completed > 0 else 0
                    avg_ms = sum(times) / len(times) if times else 0
                    print(
                        f"── Progress: {completed}/{to_process} ({completed*100/to_process:.1f}%) "
                        f"| Elapsed: {_format_time(wall)} "
                        f"| ETA: ~{_format_time(eta)} "
                        f"| Avg: {avg_ms/1000:.1f}s/q ──",
                        flush=True,
                    )

        async def process_with_semaphore(idx: int, item: dict) -> PipelineResult:
            async with semaphore:
                try:
                    result = await self.process_question(
                        question=item["question"],
                        db_id=item["db_id"],
                        evidence=item.get("evidence", ""),
                        db_root=db_root,
                    )
                except Exception as e:
                    logger.error(f"[{idx}] Failed: {e}")
                    result = PipelineResult(
                        question=item["question"],
                        db_id=item["db_id"],
                        predicted_sql="SELECT 1",
                        candidates_count=0,
                        unique_results_count=0,
                        selected_by="error",
                        processing_time_ms=0,
                        metadata={"error": str(e)},
                    )
                await _on_complete(idx, item, result)
                return result

        # Launch all tasks
        print(f"\nProcessing {to_process} questions (max_concurrent={max_concurrent})...\n", flush=True)
        tasks = []
        for idx in range(resume_from, total):
            tasks.append(process_with_semaphore(idx, data[idx]))

        new_results = await asyncio.gather(*tasks)
        results.extend(new_results)

        # Save results
        self._save_results(results, data, output_path)
        return results

    def _save_results(
        self,
        results: list[PipelineResult],
        data: list[dict],
        output_path: Path,
    ) -> None:
        """Save results in both JSON and BIRD submission format."""
        # JSON format (detailed)
        json_results = []
        for r in results:
            json_results.append({
                "question": r.question,
                "db_id": r.db_id,
                "predicted_sql": r.predicted_sql,
                "candidates_count": r.candidates_count,
                "unique_results_count": r.unique_results_count,
                "selected_by": r.selected_by,
                "processing_time_ms": r.processing_time_ms,
                "metadata": r.metadata,
            })

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        # BIRD submission format
        submission_path = output_path.with_suffix(".txt")
        with open(submission_path, "w") as f:
            for r in results:
                # Flatten SQL to single line for BIRD submission format
                flat_sql = " ".join(r.predicted_sql.split())
                f.write(f"{flat_sql}\t----- bird -----\t{r.db_id}\n")

        logger.info(f"Saved {len(results)} results to {output_path} and {submission_path}")
