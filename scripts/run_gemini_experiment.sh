#!/bin/bash
# Run language exploration with Gemini VLM
#
# Requires GEMINI_API_KEY environment variable to be set
#
# Usage:
#   export GEMINI_API_KEY="your-api-key"
#   ./scripts/run_gemini_experiment.sh
#   ./scripts/run_gemini_experiment.sh --tasks "CloseDrawer OpenDrawer"

set -e

# Check for API key
if [ -z "${GEMINI_API_KEY}" ]; then
    echo "ERROR: GEMINI_API_KEY environment variable is not set"
    echo "Please run: export GEMINI_API_KEY='your-api-key'"
    exit 1
fi

# Default configuration
SEEDS=3
MAX_ITERATIONS=10
TASKS="CloseDrawer OpenDrawer CloseSingleDoor OpenSingleDoor"
THINKING_BUDGET=1024
VLM_MODEL="gemini-2.0-flash"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --thinking_budget)
            THINKING_BUDGET="$2"
            shift 2
            ;;
        --vlm_model)
            VLM_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Gemini Language Exploration Experiment"
echo "=============================================="
echo "VLM Model: ${VLM_MODEL}"
echo "Thinking Budget: ${THINKING_BUDGET}"
echo "Tasks: ${TASKS}"
echo "Seeds: ${SEEDS}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "=============================================="

# Run with Gemini VLM
./scripts/run_language_exploration_experiment.sh \
    --vlm gemini \
    --tasks "${TASKS}" \
    --seeds "${SEEDS}" \
    --max_iterations "${MAX_ITERATIONS}"
