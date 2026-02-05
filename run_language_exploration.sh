#!/bin/bash
# Run language exploration with automatic .env loading
#
# Usage:
#   ./run_language_exploration.sh                          # defaults: CloseDrawer, gemini, 10 iterations
#   ./run_language_exploration.sh --task OpenDrawer        # specify task
#   ./run_language_exploration.sh --vlm mock               # use mock VLM (no API)
#   ./run_language_exploration.sh --max_iterations 5       # limit iterations
proj_name=DSRL_pi0_Libero
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export PYTHONPATH="$(pwd)/robosuite:$(pwd)/robocasa:$(pwd)/openpi/src:${PYTHONPATH}"                                                
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export  OPENPI_DATA_HOME=/mmfs1/gscratch/weirdlab/dg20/dsrl_pi0/openpi_data

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading .env file..."
    set -a  # automatically export all variables
    source .env
    set +a
    echo "GEMINI_API_KEY loaded"
fi

# Activate conda environment
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate robocasa_pi0
    echo "Activated robocasa_pi0 environment"
fi

# Default arguments
TASK="TurnSinkSpout"
VLM="baseline"
MAX_ITERATIONS=10
SEED=0

# Parse arguments - pass through to Python script
ARGS=""
while [[ $# -gt 0 ]]; do
    ARGS="$ARGS $1"
    shift
done

# If no args provided, use defaults
if [ -z "$ARGS" ]; then
    ARGS="--task $TASK --vlm $VLM --max_iterations $MAX_ITERATIONS --seed $SEED --pi05"
fi

echo ""
echo "=============================================="
echo "Running Language Exploration"
echo "=============================================="
echo "Arguments: $ARGS"
echo "=============================================="
echo ""

python language_exploration.py $ARGS
