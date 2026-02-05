#!/bin/bash
# Run a single language exploration experiment
#
# Usage:
#   ./scripts/run_language_exploration.sh CloseDrawer mock 0
#   ./scripts/run_language_exploration.sh OpenDrawer gemini 42

set -e

TASK=${1:-CloseDrawer}
VLM=${2:-mock}
SEED=${3:-0}
MAX_ITERATIONS=${4:-10}
MAX_STEPS=${5:-500}
CHECKPOINT=${6:-"gs://openpi-assets/checkpoints/pi05_libero"}

echo "=============================================="
echo "Language Exploration Experiment"
echo "=============================================="
echo "Task: ${TASK}"
echo "VLM: ${VLM}"
echo "Seed: ${SEED}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Max steps: ${MAX_STEPS}"
echo "Checkpoint: ${CHECKPOINT}"
echo "=============================================="

# Activate conda environment if available
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate robocasa_pi0 2>/dev/null || true
fi

# Run the exploration
python language_exploration.py \
    --task "${TASK}" \
    --vlm "${VLM}" \
    --seed "${SEED}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --max_steps "${MAX_STEPS}" \
    --checkpoint "${CHECKPOINT}" \
    --pi05

echo ""
echo "Experiment complete!"
