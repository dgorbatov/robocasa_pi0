#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate robocasa_pi0

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
# python run_rollout.py --checkpoint RLinf-Pi0-RoboCasa/  

# Note: GCS checkpoints (gs://openpi-assets/...) only contain JAX weights.
# For PyTorch inference, use a checkpoint with model.safetensors
python run_rollout.py --checkpoint gs://openpi-assets/checkpoints/pi05_libero --policy libero --pi05