#!/usr/bin/env bash
# Run BIRD evaluation (EX and R-VES)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default paths
PREDICTED_SQL="${1:-$PROJECT_DIR/data/results/mini_dev_results.txt}"
GROUND_TRUTH="${2:-$PROJECT_DIR/data/mini_dev/mini_dev_sqlite_gold.sql}"
DB_ROOT="${3:-$PROJECT_DIR/data/mini_dev/databases}"
DIFF_JSON="${4:-$PROJECT_DIR/data/mini_dev/mini_dev_sqlite.json}"
NUM_CPUS="${5:-8}"
TIMEOUT="${6:-30}"

OUTPUT_LOG="$PROJECT_DIR/data/results/evaluation_log.json"

echo "=== BIRD Evaluation ==="
echo "Predictions: $PREDICTED_SQL"
echo "Gold:        $GROUND_TRUTH"
echo "DB Root:     $DB_ROOT"
echo ""

echo "--- Execution Accuracy (EX) ---"
python3 "$SCRIPT_DIR/evaluation_ex.py" \
    --predicted_sql_path "$PREDICTED_SQL" \
    --ground_truth_path "$GROUND_TRUTH" \
    --db_root_path "$DB_ROOT" \
    --diff_json_path "$DIFF_JSON" \
    --num_cpus "$NUM_CPUS" \
    --meta_time_out "$TIMEOUT" \
    --output_log_path "$OUTPUT_LOG"

echo ""
echo "Evaluation log saved to: $OUTPUT_LOG"
