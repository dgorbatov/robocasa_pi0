# Pi0 LIBERO RoboCasa Integration

## Current State (2026-02-03)

The `pi0_libero_model.py` now uses `create_trained_policy()` from openpi infrastructure to automatically handle both JAX/Orbax and PyTorch/safetensors checkpoints.

### How It Works

The refactored code uses openpi's `create_trained_policy()` which:
1. Downloads GCS checkpoints automatically via `maybe_download()`
2. Detects checkpoint type (JAX vs PyTorch) by checking for `model.safetensors`
3. Loads JAX checkpoints with `restore_params()` (handles Orbax/OCDBT format)
4. Loads PyTorch checkpoints with safetensors
5. Sets up all transforms correctly (LiberoInputs, LiberoOutputs, normalization)

### Checkpoint Formats

| Checkpoint | Format | Auto-Detected |
|------------|--------|---------------|
| `gs://openpi-assets/checkpoints/pi05_libero` | JAX/Orbax | Yes - uses JAX inference |
| `RLinf-Pi05-LIBERO-SFT/` (with model.safetensors) | PyTorch | Yes - uses PyTorch inference |

### Usage

```bash
# Activate environment
conda activate robocasa_pi0

# Run with GCS checkpoint (JAX format - auto-detected)
python run_rollout.py --checkpoint gs://openpi-assets/checkpoints/pi05_libero --policy libero --pi05

# Run with local PyTorch checkpoint (safetensors format - auto-detected)
python run_rollout.py --checkpoint RLinf-Pi05-LIBERO-SFT/ --policy libero --pi05

# Or use the convenience script
./run_eval.sh
```

### Key Files

1. **`pi0_libero_model.py`** - Wrapper using `create_trained_policy()` for checkpoint loading
2. **`openpi/src/openpi/policies/policy_config.py`** - `create_trained_policy()` implementation
3. **`openpi/src/openpi/training/config.py`** - `get_config()` for training configs
4. **`openpi/src/openpi/models/model.py`** - `restore_params()` for JAX checkpoint loading

### Environment Requirements

**Critical: numpy version compatibility**
- JAX 0.9.0 requires `numpy >= 2.0`
- Numba requires `numpy < 2.4`
- Solution: `pip install "numpy>=2.0,<2.4"` (numpy 2.3.x works)

Current working versions:
- numpy 2.3.5
- jax 0.9.0
- jaxlib 0.9.0

### Error Handling

The code includes JAX/numpy compatibility checks. If the checkpoint only has JAX weights but JAX isn't working, you'll get a helpful error message with fix options:
1. Upgrade numpy: `pip install 'numpy>=2.0'`
2. Downgrade JAX: `pip install 'jax[cpu]==0.4.35' 'jaxlib==0.4.35'`
3. Use a checkpoint with `model.safetensors` for PyTorch inference

### Notes
- Pi05 uses quantile normalization, Pi0 uses z-score (handled automatically by config)
- RoboCasa observations need 180-degree image rotation for LIBERO format
- LIBERO outputs 7D actions, padded to 12D for RoboCasa PandaOmron
- JAX inference falls back to CPU if jaxlib is not CUDA-enabled
- Warning about `os.fork()` with JAX is normal for subprocess environments

### Troubleshooting

**EGL rendering errors**: If you see `Cannot initialize a EGL device display`, this is an infrastructure issue with the compute node's EGL setup, not the model code. Check:
- `MUJOCO_GL=egl` is set
- `MUJOCO_EGL_DEVICE_ID` points to a valid GPU
- The node has proper NVIDIA EGL drivers installed

---

## Language-Based Exploration (2026-02-04)

Implements iterative goal refinement using VLM (Vision Language Model) feedback, based on [mateoguaman/dsrl_pi0](https://github.com/mateoguaman/dsrl_pi0/commit/f3d9e9e5d4368fadd823beb90a7b35a0636077d1).

### Algorithm

```
for iteration 1 to K:
  trajectory <- rollout policy with current goal g
  if success: break
  feedback <- VLM(trajectory)
  history <- history + {goal, feedback}
  g <- VLM.generate_goal(history)
```

### Key Files

| File | Description |
|------|-------------|
| `vlm_interface.py` | VLM abstraction with MockVLM, GeminiRoboticsER, DebugVLM, BaselineVLM |
| `language_exploration.py` | Main exploration script |
| `language_exploration_utils.py` | Trajectory collection and VLM formatting utilities |
| `aggregate_results.py` | Results aggregation and live monitoring |
| `scripts/run_language_exploration.sh` | Single experiment runner |
| `scripts/run_language_exploration_experiment.sh` | Multi-task, multi-seed systematic evaluation |
| `scripts/run_gemini_experiment.sh` | Real Gemini VLM evaluation |

### VLM Backends

| Backend | Description | Use Case |
|---------|-------------|----------|
| `mock` | 50 hardcoded refinement strategies | Testing pipeline without API |
| `debug` | Manual human input | Prompt engineering, debugging |
| `gemini` | Gemini Robotics-ER integration | Real VLM evaluation |
| `baseline` | Returns unchanged goals | Control condition |

### Usage

```bash
# Test with MockVLM (no API required)
python language_exploration.py --task CloseDrawer --vlm mock --max_iterations 5 --pi05

# With Gemini (requires API key)
export GEMINI_API_KEY="your-key"
python language_exploration.py --task CloseDrawer --vlm gemini --max_iterations 10 --pi05

# Run systematic experiment
./scripts/run_language_exploration_experiment.sh --vlm mock --seeds 3 --tasks "CloseDrawer OpenDrawer"

# Monitor results in real-time
python aggregate_results.py --results_dir ./logs/language_exploration --watch
```

### Dependencies

```bash
# For Gemini VLM
pip install google-genai>=0.4.0

# For video encoding (usually already present)
pip install imageio[ffmpeg]
```

### Output Structure

```
logs/language_exploration/
├── CloseDrawer_mock_seed0_20260204_120000/
│   ├── config.json           # Experiment configuration
│   ├── results.json          # Final results with history
│   ├── iteration_01_fail.mp4 # Video per iteration
│   ├── iteration_02_fail.mp4
│   └── iteration_03_success.mp4
└── ...
```

### Notes

- Uses existing `Pi0LiberoModel`, `SubprocessEnv`, and `TASK_DESCRIPTIONS`
- Each iteration creates a fresh environment to ensure independence
- Videos are saved per iteration for debugging
- VLM inputs can be logged for debugging with `log_dir` parameter
