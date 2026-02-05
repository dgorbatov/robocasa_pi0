#!/bin/bash
# Run systematic language exploration experiments across tasks and seeds
#
# Usage:
#   ./scripts/run_language_exploration_experiment.sh
#   ./scripts/run_language_exploration_experiment.sh --vlm gemini --seeds 5
#   ./scripts/run_language_exploration_experiment.sh --tasks "CloseDrawer OpenDrawer" --seeds 3

set -e

# Default configuration
VLM="mock"
SEEDS=3
MAX_ITERATIONS=10
MAX_STEPS=500
CHECKPOINT="gs://openpi-assets/checkpoints/pi05_libero"
OUTPUT_DIR="./logs/language_exploration_experiment"

# Default tasks (subset of RoboCasa atomic tasks)
TASKS="CloseDrawer OpenDrawer CloseSingleDoor OpenSingleDoor TurnOnStove TurnOffStove"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vlm)
            VLM="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --max_iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="${OUTPUT_DIR}/${VLM}_${TIMESTAMP}"
mkdir -p "${EXP_DIR}"

echo "=============================================="
echo "Language Exploration Systematic Experiment"
echo "=============================================="
echo "VLM: ${VLM}"
echo "Tasks: ${TASKS}"
echo "Seeds per task: ${SEEDS}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Max steps: ${MAX_STEPS}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output directory: ${EXP_DIR}"
echo "=============================================="

# Save experiment configuration
cat > "${EXP_DIR}/experiment_config.json" << EOF
{
    "vlm": "${VLM}",
    "tasks": "${TASKS}",
    "seeds": ${SEEDS},
    "max_iterations": ${MAX_ITERATIONS},
    "max_steps": ${MAX_STEPS},
    "checkpoint": "${CHECKPOINT}",
    "timestamp": "${TIMESTAMP}"
}
EOF

# Activate conda environment if available
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate robocasa_pi0 2>/dev/null || true
fi

# Run experiments
TOTAL=0
SUCCESS=0

for TASK in ${TASKS}; do
    for SEED in $(seq 0 $((SEEDS - 1))); do
        TOTAL=$((TOTAL + 1))
        echo ""
        echo "=============================================="
        echo "Task: ${TASK}, Seed: ${SEED}"
        echo "=============================================="

        python language_exploration.py \
            --task "${TASK}" \
            --vlm "${VLM}" \
            --seed "${SEED}" \
            --max_iterations "${MAX_ITERATIONS}" \
            --max_steps "${MAX_STEPS}" \
            --checkpoint "${CHECKPOINT}" \
            --output_dir "${EXP_DIR}" \
            --pi05 \
            && SUCCESS=$((SUCCESS + 1)) \
            || echo "WARNING: Experiment failed for ${TASK} seed ${SEED}"
    done
done

echo ""
echo "=============================================="
echo "Experiment Complete"
echo "=============================================="
echo "Total runs: ${TOTAL}"
echo "Successful runs: ${SUCCESS}"
echo "Output directory: ${EXP_DIR}"
echo ""
echo "Aggregate results with:"
echo "  python aggregate_results.py --results_dir ${EXP_DIR}"
