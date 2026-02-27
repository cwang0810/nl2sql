#!/usr/bin/env bash
# Download BIRD benchmark data (mini-dev and dev set)
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
echo "Data directory: $DATA_DIR"

# ─── Mini-Dev (from GitHub) ───
echo "=== Downloading BIRD Mini-Dev ==="
MINI_DEV_DIR="$DATA_DIR/mini_dev"
mkdir -p "$MINI_DEV_DIR"

if [ ! -d "$MINI_DEV_DIR/mini_dev_repo" ]; then
    git clone --depth 1 https://github.com/bird-bench/mini_dev.git "$MINI_DEV_DIR/mini_dev_repo"
else
    echo "Mini-dev repo already exists, skipping clone"
fi

# Copy relevant files
cp -n "$MINI_DEV_DIR/mini_dev_repo/mini_dev_data/mini_dev_sqlite.json" "$MINI_DEV_DIR/" 2>/dev/null || true

echo ""
echo "=== Downloading Mini-Dev Databases ==="
echo "Mini-dev databases need to be downloaded from HuggingFace or the BIRD website."
echo "Please run:"
echo "  pip install huggingface_hub"
echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='birdsql/bird-minidev-sqlite', repo_type='dataset', local_dir='$MINI_DEV_DIR/databases')\""
echo ""

# ─── BIRD Dev Set ───
echo "=== BIRD Dev Set ==="
DEV_DIR="$DATA_DIR/bird_dev"
mkdir -p "$DEV_DIR"
echo "The BIRD dev set (with 95 databases, ~33.4GB) needs to be downloaded from:"
echo "  https://bird-bench.github.io/"
echo ""
echo "After downloading, extract to: $DEV_DIR/"
echo "Expected structure:"
echo "  $DEV_DIR/dev.json"
echo "  $DEV_DIR/dev_gold.sql"
echo "  $DEV_DIR/dev_databases/{db_name}/{db_name}.sqlite"
echo ""

# ─── BIRD Training Set ───
echo "=== BIRD Training Set ==="
TRAIN_DIR="$DATA_DIR/bird_train"
mkdir -p "$TRAIN_DIR"
echo "Download training set from the BIRD website for few-shot example index."
echo "Expected: $TRAIN_DIR/train.json"
echo ""

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Download databases (see instructions above)"
echo "  2. Run: python scripts/build_index.py"
echo "  3. Run: python scripts/run_mini_dev.py"
