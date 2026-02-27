# BIRD-SQL NL2SQL 打榜方案

## 快速开始

### 1. 环境配置

```bash
cd nl2sql
pip install -r requirements.txt
```

### 2. 配置 API Key（阿里百炼 DashScope）

两个模型均通过阿里百炼调用，只需一个 Key：

```bash
cp .env.example .env
# 编辑 .env，填入真实 Key
# DASHSCOPE_API_KEY=your-dashscope-api-key
```

程序会自动加载项目根目录的 `.env`，也支持你手动 `export DASHSCOPE_API_KEY=...`。

默认模型配置（可在 config.yaml 中修改）：
- DeepSeek V3: `deepseek-v3`
- Qwen3 Coder: `qwen-coder-plus-latest`

### 3. 下载数据

```bash
bash scripts/setup_data.sh
```

按提示下载 BIRD 数据集（mini-dev + dev + train）。

### 4. 构建索引

```bash
python scripts/build_index.py --train_json data/bird_train/train.json
```

### 5. 运行评测

```bash
# Mini-dev 快速迭代（500 题）
python scripts/run_mini_dev.py --limit 10  # 先测 10 题

# Mini-dev 全量
python scripts/run_mini_dev.py

# 评测
bash evaluation/run_evaluation.sh
```

### 6. 错误分析

```bash
python scripts/analyze_errors.py \
    --predictions data/results/mini_dev_results.json \
    --gold data/mini_dev/mini_dev_sqlite_gold.sql \
    --data_json data/mini_dev/mini_dev_sqlite.json \
    --db_root data/mini_dev/databases \
    --output data/results/error_analysis.json
```

### 7. 生成提交文件

```bash
python scripts/submit_test.py \
    --data_json data/bird_test/test.json \
    --db_root data/bird_test/test_databases
```

提交至 bird.bench23@gmail.com。

---

## 架构概览

三阶段 Agentic Pipeline，借鉴 Agentar-Scale-SQL 的"苦涩教训"——推理时计算规模化取胜。

```
Stage 1: 任务理解
  Question → Schema Linking → Value Retrieval → Evidence → Few-shot

Stage 2: 多路 SQL 候选生成（并行）
  ICL Generator (DeepSeek) × 8 + CoT Generator (Qwen3) × 8
  → SQL Fixer + SQL Revisor → 16~20 refined candidates

Stage 3: SQL 智能选择
  执行去重 → 锦标赛两两对抗 → 最终 SQL
```

## 核心模型

- **DeepSeek V3.2**: ICL 生成器主力模型
- **Qwen3 480B Coder**: CoT 推理生成器 + 锦标赛裁判

## 目标

- BIRD Dev EX > 75%（当前 SOTA: 81.95%）
- BIRD Test EX > 80%

## 项目结构

```
nl2sql/
├── config/config.yaml          # 配置文件
├── src/
│   ├── pipeline.py             # 主 Pipeline
│   ├── models/                 # LLM 客户端
│   ├── stage1_understanding/   # 任务理解模块
│   ├── stage2_generation/      # SQL 生成模块
│   ├── stage3_selection/       # SQL 选择模块
│   └── utils/                  # 工具类
├── scripts/                    # 运行脚本
├── evaluation/                 # 评测代码
└── data/                       # 数据目录
```
