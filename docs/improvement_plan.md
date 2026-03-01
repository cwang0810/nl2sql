# NL2SQL Pipeline 改进方案

## 当前基线

- **Mini-dev 500题 EX**: 63.8% (319/500)
  - Simple: 77.0% (114/148)
  - Moderate: 61.2% (153/250)
  - Challenging: 51.0% (52/102)

## 错误分析关键发现

| 发现 | 数据 | 影响 |
|------|------|------|
| Tournament选择器灾难性失败 | 2/34 = 5.9% | 直接损失 ~32 题 |
| Fixer/Revisor帮倒忙 | Fixer 12.5%, CoT+Revisor 0% | 修复后更差 |
| 108道错题中正确候选可能存在 | unique≥3时错误率高 | 选择机制是最大瓶颈 |
| ICLGenerator远优于CoTGenerator | 69% vs 59.6% | CoT策略需优化 |
| Direct策略最优 | 70.3% vs ~57% | 其他策略拖后腿 |
| 低温度更好 | temp=0.0: 74.2% | 高温引入噪声 |
| 12道浮点精度错误 | 评测时数值微小差异 | 容易修复 |
| 最差数据库 | financial 40.6%, toxicology 45% | 需要针对性优化 |

## 预估提升空间

- 修复Tournament选择器: +15~20题
- 修复浮点精度问题: +12题
- 改进Selection机制（108道可能有正确候选）: +20~30题
- 改进Schema Linking: +10~15题
- 改进Generation多样性: +10~15题
- **总计预估**: 67~92题 → 目标 75%~82%

---

## 改进方案（按优先级排序）

### P0: 紧急修复（预估 +30题，达到 ~70%）

#### 1. 修复 Tournament 选择器
**问题**: 当前 tournament 仅 5.9% 准确率，34次使用中只对了2次。
**原因**: tournament 在 unique≥4 的高分歧场景下被触发，而这些场景本身就很难。当前实现可能存在 position bias 消除不彻底、判断 prompt 质量差等问题。

**方案**:
- 将 tournament 触发条件从 `len(deduped) <= 3` 改为仅在 `len(deduped) == 2` 时使用
- 当 unique≥3 时，直接使用改进后的 self_consistency（加权投票）
- 改进 tournament prompt：加入 schema 上下文、evidence、执行结果对比

#### 2. 修复浮点精度评测问题
**问题**: 12道题因浮点精度微小差异被判错。
**方案**: 在评测脚本中加入数值容差比较（相对误差 < 1e-4 视为正确）

#### 3. 禁用/改进 Fixer 和 Revisor
**问题**: Fixer 12.5%，Revisor 0%~47%，修复后的候选在投票中权重应极低。
**方案**:
- 短期：将 fixer/revisor 产生的候选标记为低置信度，在 self_consistency 中给予极低权重（0.01）
- 中期：改进 Revisor 触发条件，不仅仅是 empty result，还要检查结果合理性

---

### P1: Schema Linking 增强（预估 +10~15题）

#### 4. 双向 Schema Linking（借鉴 RSL-SQL）
**现状**: 单向 embedding top-30 → LLM 精炼
**改进**:
- **正向剪枝**: question → 提取关键词/实体 → 匹配 table/column（当前已有）
- **反向剪枝**: 对 LLM 精炼结果，反向验证每个选中的 table/column 是否真的与问题相关
- **双模式投票**: 同时生成 "精简schema" 和 "完整schema" 两个版本的 SQL，最终投票选择

**实现**: 新增 `bidirectional_schema_linker.py`，在现有 SchemaLinker 基础上增加反向验证步骤

#### 5. 问题分解辅助 Schema Linking
**现状**: 复杂多表查询时 schema linking 容易遗漏关键表
**改进**: 对 moderate/challenging 问题，先用 LLM 将问题分解为子问题，每个子问题独立做 schema linking，然后合并结果
**实现**: 新增 `question_decomposer.py`

---

### P2: SQL Generation 多样性增强（预估 +10~15题）

#### 6. 在线示例合成（借鉴 CHASE-SQL）
**现状**: 从训练集检索 top-3 few-shot 示例
**改进**: 除了检索示例外，针对当前问题在线合成 1-2 个定制化示例
- 基于当前 schema 和问题类型，让 LLM 生成一个类似但更简单的 question-SQL 对
- 作为额外的 few-shot 示例注入 prompt

**实现**: 新增 `online_example_synthesizer.py`

#### 7. 子查询分解生成（借鉴 CHASE-SQL Divide-and-Conquer）
**现状**: divide_conquer prompt 只是提示 LLM 分步思考
**改进**: 真正的分解执行 ——
- 将复杂问题分解为 2-3 个子问题
- 每个子问题独立生成 sub-SQL
- 用 LLM 将 sub-SQL 组合为最终 SQL
- 作为额外的候选加入投票

**实现**: 新增 `decompose_generator.py`

#### 8. 调整温度和策略分配
**现状**: 4温度 × 2策略 = 8候选/生成器
**改进**: 基于数据分析优化分配
- ICL: temp=[0.0, 0.0, 0.3, 0.5] × prompts=["direct", "direct", "cot"] = 更多低温+direct
- CoT: temp=[0.0, 0.0, 0.3, 0.5] × prompts=["step_by_step", "divide_conquer"]
- 新增: decompose_generator 产生 2-4 个额外候选
- 总候选数: ~24个（从16增加到24，适度增加）

---

### P3: Selection 机制重构（预估 +20~30题）

#### 9. 执行结果感知的加权投票
**现状**: self_consistency 基于经验权重
**改进**:
- 加入执行结果质量信号：非空结果 > 空结果，合理行数 > 异常行数
- 加入 schema 覆盖度信号：SQL 中使用的表/列是否都在 linked schema 中
- 动态权重：根据当前问题的 unique_results_count 调整策略

#### 10. 多轮投票 + 执行验证
**现状**: 一次性投票选择
**改进**:
- 第一轮：按执行结果分组，加权投票选出 top-2 候选
- 第二轮：对 top-2 候选做深度 LLM 对比（带完整 schema + evidence + 执行结果）
- 如果两轮结果一致，高置信度输出；否则选择第一轮结果

#### 11. 执行结果交叉验证
**新增**: 对最终选出的 SQL，做一次 "sanity check"
- 检查结果行数是否合理（问 "how many" 应返回1行）
- 检查结果类型是否匹配（问 "who" 应返回字符串）
- 如果不通过，回退到第二候选

---

### P4: 跨阶段优化

#### 12. 自适应 Schema 模式
- 简单问题（单表）：使用精简 schema，减少噪声
- 复杂问题（多表 JOIN）：使用完整 schema + 问题分解
- 判断依据：问题中的关键词（JOIN、多个表名、子查询暗示）

#### 13. 数据库特定优化
- 对 financial、toxicology、california_schools 等低准确率数据库
- 分析错误模式，添加针对性的 domain hints
- 在 `config/domain_hints.yaml` 中补充更多数据库特定规则

---

## 实施路线图

### Phase 1: 紧急修复（1-2天，预估 63.8% → ~70%）
1. 修复 Tournament 选择器逻辑
2. 修复浮点精度评测
3. 降低 Fixer/Revisor 权重

### Phase 2: Schema Linking + Selection 重构（3-5天，预估 ~70% → ~75%）
4. 双向 Schema Linking
5. 问题分解辅助 Schema Linking
9. 执行结果感知的加权投票
10. 多轮投票 + 执行验证

### Phase 3: Generation 增强（3-5天，预估 ~75% → ~78%+）
6. 在线示例合成
7. 子查询分解生成
8. 温度和策略优化

### Phase 4: 精细调优（2-3天，预估 ~78% → ~80%+）
11. 执行结果交叉验证
12. 自适应 Schema 模式
13. 数据库特定优化

---

## 参考文献

- [Agentar-Scale-SQL](https://arxiv.org/abs/2509.24403) — BIRD Test 81.67%, #1 on leaderboard
- [CHASE-SQL](https://vercel.hyper.ai/en/papers/2410.01943) — Multi-path reasoning + preference-optimized selection, BIRD Test 73%
- [RSL-SQL](https://arxiv.org/html/2411.00073v2) — Bidirectional schema linking, 94% recall with 83% column reduction
- [Databricks RLVR](https://www.databricks.com/blog/power-rlvr-training-leading-sql-reasoning-model-databricks) — RLVR-trained SQL reasoning model, top single-model on BIRD
- [CHESS](https://openreview.net/forum?id=fObObsreRH) — 4-agent pipeline: retriever + schema selector + generator + unit tester
- [MAG-SQL](https://www.aimodels.fyi/papers/arxiv/mag-sql-multi-agent-generative-approach-soft) — Multi-agent with soft schema linking + iterative sub-SQL refinement
- [Knapsack Schema Linking](https://arxiv.org/html/2502.12911v1) — Optimization-based schema element selection
