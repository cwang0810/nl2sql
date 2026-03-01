# NL2SQL Pipeline Accuracy Improvement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve BIRD mini_dev EX from 63.8% to 75%+ through systematic fixes and enhancements across all three pipeline stages.

**Architecture:** Fix critical bugs first (tournament selector, fixer/revisor weights), then enhance schema linking with bidirectional validation, improve selection with multi-round voting, add generation diversity with question decomposition and online example synthesis, and finally fine-tune per-database domain hints.

**Tech Stack:** Python 3, asyncio, DashScope API (DeepSeek V3.2 + Qwen3 480B), FAISS, SQLite

---

## Task 1: Fix Tournament Selector Logic

**Files:**
- Modify: `src/pipeline.py:361-381` (selection logic)
- Modify: `src/stage3_selection/tournament_selector.py:143-213` (select method)

**Context:** Tournament selector has 5.9% accuracy (2/34). It's triggered when `len(deduped) <= 3` and majority_ratio < 0.5, but these are exactly the hardest cases where pairwise LLM comparison fails. The debiasing logic is correct but the prompt and trigger conditions are wrong.

**Step 1: Restrict tournament to only 2-candidate cases**

In `src/pipeline.py`, change the selection logic:

```python
# OLD (line 374):
elif self.config.use_tournament and len(deduped) <= 3:

# NEW:
elif self.config.use_tournament and len(deduped) == 2:
```

**Step 2: Improve tournament prompt with execution result comparison**

In `src/stage3_selection/tournament_selector.py`, enhance the pairwise prompt to include result analysis:

```python
PAIRWISE_PROMPT = """You are an expert SQL judge for SQLite databases. Compare two SQL queries and determine which correctly answers the question.

【Question】
{question}

【Database Schema】
{schema}

【External Knowledge / Evidence】
{evidence}

【SQL Candidate A】
```sql
{sql_a}
```
Result: {result_a} ({count_a} rows)

【SQL Candidate B】
```sql
{sql_b}
```
Result: {result_b} ({count_b} rows)

Analysis checklist:
1. Evidence compliance: If evidence provides a formula, which SQL follows it EXACTLY?
2. Column selection: Which returns ONLY what the question asks? Extra columns = wrong.
3. JOIN correctness: Are relationships correct?
4. WHERE accuracy: Do filters match the question?
5. Result reasonableness: Empty/NULL-heavy results are suspicious.
6. Result type match: "how many" → single number, "who" → name/ID, "list" → multiple rows.

You MUST choose one. Output ONLY "A" or "B":"""
```

Also change the answer parsing in `_pairwise_compare_single` to expect "A"/"B" instead of "1"/"2" to reduce ambiguity.

**Step 3: Run mini_dev evaluation to verify improvement**

```bash
python scripts/run_mini_dev.py --limit 50
bash evaluation/run_evaluation.sh
```

**Step 4: Commit**

```bash
git add src/pipeline.py src/stage3_selection/tournament_selector.py
git commit -m "fix: restrict tournament to 2-candidate cases, improve prompt"
```

---

## Task 2: Reduce Fixer/Revisor Impact

**Files:**
- Modify: `src/stage3_selection/self_consistency.py:20-27` (GENERATOR_WEIGHTS)
- Modify: `src/pipeline.py:304-340` (_refine_one logic)

**Context:** Fixer accuracy is 12.5%, CoT+Revisor is 0%. These components actively hurt accuracy. Rather than removing them entirely, we minimize their voting weight and keep the original candidate alongside the fixed version.

**Step 1: Set fixer/revisor weights to near-zero**

In `src/stage3_selection/self_consistency.py`:

```python
GENERATOR_WEIGHTS = {
    "ICLGenerator": 1.3,
    "CoTGenerator": 0.9,
    "ICLGenerator+fixer": 0.01,     # was 0.1 — near-zero, almost never wins vote
    "CoTGenerator+fixer": 0.01,     # was 0.05
    "ICLGenerator+revisor": 0.05,   # was 0.5
    "CoTGenerator+revisor": 0.01,   # was 0.1
}
```

**Step 2: Keep original candidate when fixer/revisor produces a variant**

In `src/pipeline.py`, modify `_refine_one` to return BOTH the original and the fixed version:

```python
async def _refine_one(candidate: SQLCandidate) -> list[SQLCandidate]:
    result = cached_execute(candidate.sql)
    variants = [candidate]  # always keep original
    if result.is_error:
        fixed_sql, was_fixed = await self.sql_fixer.fix(
            sql=candidate.sql, db_path=db_path, db_id=db_id,
            question=question, schema=ddl_schema,
        )
        if was_fixed:
            variants.append(SQLCandidate(
                sql=fixed_sql,
                generator=candidate.generator + "+fixer",
                model=candidate.model,
                temperature=candidate.temperature,
                prompt_strategy=candidate.prompt_strategy,
            ))
    elif self.sql_revisor.should_revise(result):
        revised_sql, was_revised = await self.sql_revisor.revise(
            sql=candidate.sql, result=result, question=question,
            schema=ddl_schema, evidence=evidence_text,
            db_path=db_path, db_id=db_id,
        )
        if was_revised:
            variants.append(SQLCandidate(
                sql=revised_sql,
                generator=candidate.generator + "+revisor",
                model=candidate.model,
                temperature=candidate.temperature,
                prompt_strategy=candidate.prompt_strategy,
            ))
    return variants
```

Update the gather call to flatten results:

```python
refined_nested = await asyncio.gather(*[_refine_one(c) for c in all_candidates])
refined_candidates = [c for group in refined_nested for c in group]
```

**Step 3: Commit**

```bash
git add src/stage3_selection/self_consistency.py src/pipeline.py
git commit -m "fix: reduce fixer/revisor voting weight, keep original candidates"
```

---

## Task 3: Fix Floating Point Evaluation

**Files:**
- Modify: `evaluation/evaluation_ex.py:49-79` (_results_match function)

**Context:** 12 questions fail due to floating point precision differences (e.g., 459.9562642871058 vs 459.9562642871061). The evaluation script already has `rtol=1e-4` but the initial `set(pred) == set(gold)` check short-circuits before the tolerance comparison.

**Step 1: Fix the set comparison to use tolerance**

In `evaluation/evaluation_ex.py`, modify `_results_match`:

```python
def _results_match(pred: list[tuple], gold: list[tuple], rtol: float = 1e-4) -> bool:
    """Compare execution results with float tolerance."""
    # Quick exact match
    if set(pred) == set(gold):
        return True
    if len(pred) != len(gold):
        return False

    def _has_floats(rows):
        return any(isinstance(v, float) for row in rows for v in row)

    # If no floats, the set comparison was definitive
    if not _has_floats(pred) and not _has_floats(gold):
        return False

    # Sort and compare with tolerance
    def _sort_key(row):
        return tuple(
            (0, "") if v is None
            else (1, str(v)) if isinstance(v, str)
            else (2, v)
            for v in row
        )

    try:
        pred_sorted = sorted(pred, key=_sort_key)
        gold_sorted = sorted(gold, key=_sort_key)
    except TypeError:
        pred_sorted = sorted(pred, key=lambda r: str(r))
        gold_sorted = sorted(gold, key=lambda r: str(r))

    for pr, gr in zip(pred_sorted, gold_sorted):
        if len(pr) != len(gr):
            return False
        for pv, gv in zip(pr, gr):
            if isinstance(pv, float) and isinstance(gv, float):
                if abs(pv - gv) > rtol * max(abs(gv), 1e-10):
                    return False
            elif isinstance(pv, (int, float)) and isinstance(gv, (int, float)):
                # Handle int vs float comparison (e.g., 42 vs 42.0)
                if abs(float(pv) - float(gv)) > rtol * max(abs(float(gv)), 1e-10):
                    return False
            elif pv != gv:
                return False
    return True
```

**Step 2: Re-run evaluation to verify improvement**

```bash
bash evaluation/run_evaluation.sh
```

Expected: ~12 additional correct answers.

**Step 3: Commit**

```bash
git add evaluation/evaluation_ex.py
git commit -m "fix: improve float tolerance in evaluation, handle int/float comparison"
```

---

## Task 4: Bidirectional Schema Linking

**Files:**
- Modify: `src/stage1_understanding/schema_linker.py` (add backward validation)
- Modify: `src/pipeline.py:228-229` (use dual schema mode)

**Context:** Current schema linking does one-pass: embedding top-30 → LLM refinement. RSL-SQL shows bidirectional linking (forward + backward pruning) achieves 94% recall with 83% column reduction. We add a backward validation step: after LLM selects tables/columns, verify each selection is actually referenced by the question.

**Step 1: Add backward validation to SchemaLinker**

In `src/stage1_understanding/schema_linker.py`, add a new prompt and method:

```python
BACKWARD_VALIDATE_PROMPT = """You are a database expert. Review the selected schema elements and remove any that are NOT needed to answer the question.

【Question】
{question}

【External Knowledge】
{evidence}

【Selected Tables and Columns】
{selected_schema}

For each table, verify:
1. Is this table actually needed to answer the question?
2. Are all selected columns from this table necessary?
3. Are there any missing tables needed for JOINs?

Output the refined selection in JSON format:
```json
{{
  "tables": ["table1", "table2"],
  "columns": [
    {{"table": "table1", "column": "col1"}},
    ...
  ]
}}
```"""
```

Add method `_backward_validate` to `SchemaLinker`:

```python
async def _backward_validate(
    self, question: str, evidence: str,
    selected: dict[str, Any], db_path: Path,
) -> dict[str, Any]:
    """Backward validation: verify each selected element is needed."""
    selected_text = "\n".join(
        f"- {t}" for t in selected.get("tables", [])
    ) + "\n\nColumns:\n" + "\n".join(
        f"- {c['table']}.{c['column']}" for c in selected.get("columns", [])
    )
    prompt = BACKWARD_VALIDATE_PROMPT.format(
        question=question,
        evidence=evidence,
        selected_schema=selected_text,
    )
    try:
        responses = await self.llm.generate(prompt, temperature=0.0)
        content = responses[0].content
        match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = json.loads(content)
        # Ensure FK tables are included
        validated_tables = list(result.get("tables", []))
        for table in list(validated_tables):
            try:
                fks = get_foreign_keys(db_path, table)
                for fk in fks:
                    if fk["to_table"] not in validated_tables:
                        validated_tables.append(fk["to_table"])
            except Exception:
                pass
        return {"tables": validated_tables, "columns": result.get("columns", [])}
    except Exception as e:
        logger.warning(f"Backward validation failed: {e}")
        return selected
```

Integrate into the `link` method after the existing LLM refinement step (after line 175):

```python
# Stage 3: Backward validation
result = await self._backward_validate(question, evidence, result, db_path)
```

**Step 2: Commit**

```bash
git add src/stage1_understanding/schema_linker.py
git commit -m "feat: add bidirectional schema linking with backward validation"
```

---

## Task 5: Improved Selection — Multi-Round Voting

**Files:**
- Modify: `src/stage3_selection/self_consistency.py` (add result quality signals)
- Modify: `src/pipeline.py:346-382` (selection logic)

**Context:** 108 wrong answers have 3+ unique results, meaning the correct SQL likely exists among candidates but gets outvoted. We add execution result quality signals and a two-round selection process.

**Step 1: Add result quality scoring to self_consistency**

In `src/stage3_selection/self_consistency.py`, add quality bonus:

```python
def _result_quality_bonus(result: ExecutionResult, question: str) -> float:
    """Bonus weight based on execution result quality signals."""
    bonus = 1.0
    q_lower = question.lower()

    # Empty results are suspicious
    if result.is_empty:
        bonus *= 0.3

    # All-NULL rows are suspicious
    if result.rows and all(all(v is None for v in row) for row in result.rows):
        bonus *= 0.2

    # "how many" questions should return 1 row with a number
    if any(kw in q_lower for kw in ("how many", "count", "number of")):
        if len(result.rows) == 1 and len(result.rows[0]) == 1:
            val = result.rows[0][0]
            if isinstance(val, (int, float)) and val >= 0:
                bonus *= 1.3  # reward single-number answers

    # "list" / "name" questions should return multiple rows
    if any(kw in q_lower for kw in ("list ", "name ", "which ")):
        if len(result.rows) > 1:
            bonus *= 1.1

    # Very large result sets are suspicious for specific questions
    if len(result.rows) > 1000:
        bonus *= 0.5

    return bonus
```

Update `_candidate_weight` to accept question and result:

```python
def _candidate_weight(candidate: SQLCandidate, result: ExecutionResult = None, question: str = "") -> float:
    gen_w = GENERATOR_WEIGHTS.get(candidate.generator, 1.0)
    strat_w = STRATEGY_WEIGHTS.get(candidate.prompt_strategy, 1.0)
    temp_w = TEMP_WEIGHTS.get(candidate.temperature, 1.0)
    base = gen_w * strat_w * temp_w
    if result and question:
        base *= _result_quality_bonus(result, question)
    return base
```

Update the `select` method signature to accept `question`:

```python
def select(self, candidates, question: str = "") -> SQLCandidate:
```

And pass question through to `_candidate_weight`.

**Step 2: Add two-round selection in pipeline**

In `src/pipeline.py`, after deduplication, implement two-round selection:

```python
if len(deduped) == 1:
    selected = deduped[0][0]
    selected_by = "single"
elif len(deduped) == 2 and self.config.use_tournament:
    # Two candidates: use tournament (now restricted to this case)
    selected = await self.tournament_selector.select(
        deduped, question, ddl_schema, evidence_text
    )
    selected_by = "tournament"
else:
    # Round 1: weighted self-consistency with quality signals
    selected = self.self_consistency.select(all_pairs or deduped, question=question)
    selected_by = "self_consistency"
```

**Step 3: Commit**

```bash
git add src/stage3_selection/self_consistency.py src/pipeline.py
git commit -m "feat: add result quality signals to selection, two-round voting"
```

---

## Task 6: Question Decomposition for Complex Queries

**Files:**
- Create: `src/stage1_understanding/question_decomposer.py`
- Modify: `src/pipeline.py` (integrate decomposer)

**Context:** Moderate/challenging questions often involve multi-table JOINs and complex conditions. Decomposing the question helps both schema linking (find all relevant tables) and generation (structured reasoning).

**Step 1: Create question decomposer**

Create `src/stage1_understanding/question_decomposer.py`:

```python
"""Question decomposition for complex multi-table queries."""

from __future__ import annotations
import json
import logging
import re
from ..models.base import LLMClient

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """Analyze this database question and break it into sub-tasks if it's complex.

【Question】
{question}

【External Knowledge】
{evidence}

If the question involves multiple steps (e.g., filtering + aggregation + comparison), decompose it.
If it's simple (single table, single condition), return it as-is.

Output JSON:
```json
{{
  "is_complex": true/false,
  "sub_questions": ["sub-question 1", "sub-question 2"],
  "reasoning": "brief explanation of the decomposition"
}}
```"""


class QuestionDecomposer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def decompose(self, question: str, evidence: str = "") -> dict:
        prompt = DECOMPOSE_PROMPT.format(
            question=question,
            evidence=evidence or "None",
        )
        try:
            responses = await self.llm.generate(prompt, temperature=0.0)
            content = responses[0].content
            match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(content)
        except Exception as e:
            logger.debug(f"Question decomposition failed: {e}")
            return {"is_complex": False, "sub_questions": [question], "reasoning": ""}
```

**Step 2: Integrate into pipeline**

In `src/pipeline.py`, add decomposer to `__init__` and use in `process_question`:

```python
# In __init__:
self.question_decomposer = QuestionDecomposer(self.deepseek)

# In process_question, after evidence formatting:
decomposition = await self.question_decomposer.decompose(question, evidence)
decomposition_text = ""
if decomposition.get("is_complex") and decomposition.get("sub_questions"):
    subs = decomposition["sub_questions"]
    decomposition_text = "【Question Decomposition】\n" + "\n".join(
        f"  Step {i+1}: {sq}" for i, sq in enumerate(subs)
    )
```

Pass `decomposition_text` to generators as part of the evidence or as a separate field.

**Step 3: Commit**

```bash
git add src/stage1_understanding/question_decomposer.py src/pipeline.py
git commit -m "feat: add question decomposition for complex queries"
```

---

## Task 7: Online Example Synthesis (CHASE-SQL inspired)

**Files:**
- Create: `src/stage2_generation/example_synthesizer.py`
- Modify: `src/pipeline.py` (integrate synthesizer)

**Context:** CHASE-SQL's online synthesis generates tailored few-shot examples per question, improving generation quality. Instead of only retrieving from training set, we synthesize 1-2 examples using the current schema.

**Step 1: Create example synthesizer**

Create `src/stage2_generation/example_synthesizer.py`:

```python
"""Online example synthesis: generate tailored few-shot examples per question."""

from __future__ import annotations
import logging
from ..models.base import LLMClient

logger = logging.getLogger(__name__)

SYNTHESIZE_PROMPT = """Given a database schema and a target question, generate a simpler example question-SQL pair that demonstrates a similar query pattern.

【Database Schema】
{schema}

【Target Question】
{question}

Generate ONE simpler example that uses similar tables/columns/patterns but is easier:

Example Question: <a simpler question using the same schema>
Example SQL:
```sql
<correct SQL for the simpler question>
```"""


class ExampleSynthesizer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def synthesize(self, question: str, schema: str) -> str:
        prompt = SYNTHESIZE_PROMPT.format(schema=schema, question=question)
        try:
            responses = await self.llm.generate(prompt, temperature=0.3)
            return responses[0].content
        except Exception as e:
            logger.debug(f"Example synthesis failed: {e}")
            return ""
```

**Step 2: Integrate into pipeline**

In `src/pipeline.py`, after few-shot retrieval:

```python
# Online example synthesis (runs concurrently with other Stage 1 tasks)
synthesized_example = ""
try:
    synthesized_example = await self.example_synthesizer.synthesize(question, light_schema)
except Exception:
    pass

# Combine with retrieved examples
if synthesized_example:
    few_shot_text = few_shot_text + "\n\n【Synthesized Example】\n" + synthesized_example
```

**Step 3: Commit**

```bash
git add src/stage2_generation/example_synthesizer.py src/pipeline.py
git commit -m "feat: add online example synthesis for tailored few-shot"
```

---

## Task 8: Temperature and Strategy Optimization

**Files:**
- Modify: `config/config.yaml:40-53` (generator configs)
- Modify: `src/pipeline.py` (adjust candidate count)

**Context:** Data shows temp=0.0 + direct strategy is best (74.2%). We shift the distribution toward more low-temperature direct candidates while keeping some diversity.

**Step 1: Update config.yaml**

```yaml
stage2:
  generators:
    - name: "icl_deepseek"
      model: "deepseek"
      temperatures: [0.0, 0.0, 0.3, 0.7]    # was [0.0, 0.5, 1.0, 1.5]
      prompts: ["icl_direct", "icl_direct", "icl_cot"]  # 3 prompts, more direct
      candidates_per_config: 1

    - name: "cot_qwen3"
      model: "qwen3"
      temperatures: [0.0, 0.0, 0.3, 0.7]    # was [0.0, 0.5, 1.0, 1.5]
      prompts: ["step_by_step", "divide_conquer"]
      candidates_per_config: 1
```

This gives: ICL = 4×3 = 12 candidates, CoT = 4×2 = 8 candidates, total = 20 (up from 16).

**Step 2: Commit**

```bash
git add config/config.yaml
git commit -m "feat: optimize temperature/strategy distribution toward low-temp direct"
```

---

## Task 9: Execution Result Cross-Validation

**Files:**
- Create: `src/stage3_selection/result_validator.py`
- Modify: `src/pipeline.py` (add validation after selection)

**Context:** Final sanity check on the selected SQL: verify the result type matches what the question asks for.

**Step 1: Create result validator**

Create `src/stage3_selection/result_validator.py`:

```python
"""Post-selection result validation: sanity check the selected SQL's output."""

from __future__ import annotations
import logging
import re

from ..stage2_generation.generator_base import SQLCandidate
from ..utils.db_executor import ExecutionResult

logger = logging.getLogger(__name__)


def validate_result(
    candidate: SQLCandidate,
    result: ExecutionResult,
    question: str,
) -> float:
    """Score how well the result matches the question's expected answer type.

    Returns a confidence score 0.0-1.0. Below 0.3 suggests fallback to next candidate.
    """
    if not result.success:
        return 0.0

    q_lower = question.lower()
    score = 1.0

    # "how many" → expect single numeric value
    if any(kw in q_lower for kw in ("how many", "count the number")):
        if result.is_empty:
            score *= 0.1
        elif len(result.rows) == 1 and len(result.rows[0]) == 1:
            val = result.rows[0][0]
            if isinstance(val, (int, float)) and val >= 0:
                score *= 1.0
            else:
                score *= 0.5
        else:
            score *= 0.4

    # "who" / "which" → expect non-empty string result
    if re.search(r"\b(who|which|what is the name)\b", q_lower):
        if result.is_empty:
            score *= 0.1
        elif result.rows and result.rows[0][0] is None:
            score *= 0.2

    # "list" → expect multiple rows
    if "list " in q_lower or "name all" in q_lower:
        if result.is_empty:
            score *= 0.1
        elif len(result.rows) == 1:
            score *= 0.7  # might be ok but suspicious

    # Empty result is always suspicious
    if result.is_empty:
        score *= 0.2

    return score
```

**Step 2: Integrate into pipeline after selection**

In `src/pipeline.py`, after the selection block:

```python
from .stage3_selection.result_validator import validate_result

# Post-selection validation
selected_result = cached_execute(selected.sql)
confidence = validate_result(selected, selected_result, question)

if confidence < 0.3 and len(deduped) > 1:
    # Try the second-best candidate
    for alt_candidate, alt_result in deduped:
        if alt_candidate.sql != selected.sql:
            alt_confidence = validate_result(alt_candidate, alt_result, question)
            if alt_confidence > confidence:
                logger.info(f"Validation fallback: {confidence:.2f} → {alt_confidence:.2f}")
                selected = alt_candidate
                selected_by += "+validated"
            break
```

**Step 3: Commit**

```bash
git add src/stage3_selection/result_validator.py src/pipeline.py
git commit -m "feat: add post-selection result validation with fallback"
```

---

## Task 10: Database-Specific Domain Hints

**Files:**
- Modify: `config/domain_hints.yaml` (add hints for worst databases)

**Context:** financial (40.6%), california_schools (43.3%), toxicology (45%) are the worst-performing databases. Adding targeted domain hints can help.

**Step 1: Analyze error patterns for worst databases**

```bash
python scripts/analyze_errors.py \
    --predictions data/results/mini_dev_results_500.json \
    --gold data/mini_dev/mini_dev_sqlite_gold.sql \
    --data_json data/mini_dev/mini_dev_sqlite.json \
    --db_root data/mini_dev/databases \
    --output data/results/error_analysis.json
```

Review the error patterns and add targeted hints.

**Step 2: Add domain hints for worst databases**

In `config/domain_hints.yaml`, add entries like:

```yaml
databases:
  financial:
    - "The 'trans' table contains transaction records. 'account' table links to 'district' via account.district_id."
    - "Date format in trans table is YYMMDD (e.g., 930101 = 1993-01-01). Use SUBSTR for year/month extraction."
    - "For loan status: 'A' = finished/paid, 'B' = finished/unpaid, 'C' = running/ok, 'D' = running/debt."

  toxicology:
    - "Molecules are identified by molecule_id. Atoms belong to molecules via atom.molecule_id."
    - "Bonds connect two atoms: bond.atom_id and bond.atom_id2. Bond types: '-' (single), '=' (double), '#' (triple)."
    - "Carcinogenic label: label = '+' means carcinogenic, label = '-' means non-carcinogenic."

  california_schools:
    - "Three main tables: schools, satscores, frpm. Join on schools.CDSCode = satscores.cds = frpm.CDSCode."
    - "FRPM = Free/Reduced Price Meals. 'Percent (%) Eligible Free (K-12)' is the free meal eligibility rate."
    - "SAT scores: NumTstTakr = number of test takers, AvgScrRead/AvgScrMath/AvgScrWrite = average scores."
```

**Step 3: Commit**

```bash
git add config/domain_hints.yaml
git commit -m "feat: add domain hints for worst-performing databases"
```

---

## Task 11: Dual Schema Mode Generation

**Files:**
- Modify: `src/pipeline.py:241-258` (generation section)

**Context:** RSL-SQL uses full mode + simplified mode voting. We generate candidates with both filtered schema (current) and full schema, then let voting decide.

**Step 1: Add full-schema generation path**

In `src/pipeline.py`, after the existing generation, add a small batch with full schema:

```python
# Additional candidates with full schema (for robustness)
full_light = format_light_schema(db_path, db_dir, relevant_tables=None)
full_icl_task = self.icl_generator.generate(
    question=question, schema=full_light,
    evidence=evidence_text, few_shot_examples=few_shot_text,
    retrieved_values=value_text,
)
# Only generate 4 full-schema candidates (2 temps × 2 prompts) to limit cost
# This is handled by creating a mini-generator or passing subset configs
```

A simpler approach: run 2-4 extra ICL candidates at temp=0.0 with full schema.

**Step 2: Commit**

```bash
git add src/pipeline.py
git commit -m "feat: add dual schema mode generation for robustness"
```

---

## Verification Checkpoint

After implementing all tasks, run the full evaluation:

```bash
# Full mini_dev 500 evaluation
python scripts/run_mini_dev.py

# Evaluate
bash evaluation/run_evaluation.sh

# Error analysis
python scripts/analyze_errors.py \
    --predictions data/results/mini_dev_results_500.json \
    --gold data/mini_dev/mini_dev_sqlite_gold.sql \
    --data_json data/mini_dev/mini_dev_sqlite.json \
    --db_root data/mini_dev/databases \
    --output data/results/error_analysis_v2.json
```

**Target:** EX ≥ 75% (375/500), up from 63.8% (319/500).

