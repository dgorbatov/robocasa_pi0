#!/bin/bash
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

python run_rollout.py --checkpoint RLinf-Pi0-RoboCasa/  