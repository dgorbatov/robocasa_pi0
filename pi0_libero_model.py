"""Pi0 LIBERO model wrapper for use with RoboCasa environment.

This module provides a model class that:
1. Loads LIBERO-finetuned Pi0/Pi05 checkpoints (supports both JAX/Orbax and safetensors)
2. Uses openpi's create_trained_policy() for automatic checkpoint handling
3. Converts RoboCasa observations to LIBERO format
4. Converts LIBERO 7D actions to RoboCasa 12D PandaOmron format
"""

import os
import numpy as np

from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
import openpi.shared.download as download


def _check_jax_numpy_compatibility():
    """Check if JAX and numpy versions are compatible."""
    try:
        import jax
        # Try a simple JAX operation that would fail with numpy < 2.0
        _ = jax.random.key(0)
        return True
    except TypeError as e:
        if "copy" in str(e):
            return False
        raise
    except Exception:
        return True  # Let it fail later with a proper error


def _has_safetensors(checkpoint_path: str) -> bool:
    """Check if checkpoint has safetensors weights."""
    checkpoint_dir = download.maybe_download(str(checkpoint_path))
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    return os.path.exists(weight_path)


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle representation."""
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    sin_half_angle = np.sqrt(1.0 - w * w + 1e-8)
    axis = np.stack([x, y, z], axis=-1) / (sin_half_angle[..., np.newaxis] + 1e-8)
    return axis * angle[..., np.newaxis]


class Pi0LiberoModel:
    """Pi0 LIBERO model wrapper for use with RoboCasa environment."""

    def __init__(
        self,
        checkpoint_path: str = "gs://openpi-assets/checkpoints/pi05_libero",
        device: str = "cuda",
        action_horizon: int = 10,
        num_denoise_steps: int = 5,
        pi05: bool = True,
    ):
        """Initialize the LIBERO model.

        Args:
            checkpoint_path: Path to checkpoint. Can be:
                - GCS path (e.g., "gs://openpi-assets/checkpoints/pi05_libero") - JAX/Orbax format
                - Local path with model.safetensors (e.g., "RLinf-Pi05-LIBERO-SFT/") - PyTorch format
            device: Device for inference ("cuda" or "cpu")
            action_horizon: Number of actions to predict (used for config selection)
            num_denoise_steps: Number of diffusion denoising steps
            pi05: Whether to use Pi05 model configuration (True) or Pi0 (False)
        """
        self.device = device
        self.action_horizon = action_horizon
        self.num_denoise_steps = num_denoise_steps
        self.pi05 = pi05

        # Check if checkpoint has safetensors (PyTorch) or needs JAX
        has_safetensors = _has_safetensors(checkpoint_path)

        # If no safetensors, verify JAX/numpy compatibility before attempting to load
        if not has_safetensors and not _check_jax_numpy_compatibility():
            raise RuntimeError(
                f"JAX/numpy version mismatch detected. The checkpoint at '{checkpoint_path}' "
                f"only contains JAX weights (no model.safetensors), but JAX is not working "
                f"due to numpy incompatibility.\n\n"
                f"Options to fix this:\n"
                f"  1. Upgrade numpy: pip install 'numpy>=2.0'\n"
                f"  2. Downgrade JAX: pip install 'jax[cpu]==0.4.35' 'jaxlib==0.4.35'\n"
                f"  3. Use a checkpoint with model.safetensors for PyTorch inference"
            )

        # Get training config for the appropriate model type
        config_name = "pi05_libero" if pi05 else "pi0_libero"
        train_config = get_config(config_name)

        # Create policy using openpi infrastructure
        # This handles:
        # - Downloading GCS checkpoints via maybe_download()
        # - Auto-detecting JAX vs PyTorch format
        # - Loading JAX checkpoints with restore_params() (handles Orbax/OCDBT)
        # - Loading PyTorch checkpoints with safetensors
        # - Setting up all transforms correctly (LiberoInputs, LiberoOutputs, normalization)
        self.policy = create_trained_policy(
            train_config,
            checkpoint_path,
            sample_kwargs={"num_steps": num_denoise_steps},
            pytorch_device=device,
        )

    def robocasa_obs_to_libero(self, raw_obs: dict, task_description: str) -> dict:
        """Convert RoboCasa observation to LIBERO format."""
        eef_pos = raw_obs["robot0_base_to_eef_pos"]
        eef_quat = raw_obs["robot0_base_to_eef_quat"]
        eef_axisangle = quat_to_axisangle(eef_quat)
        gripper_qpos = raw_obs["robot0_gripper_qpos"][:1]

        # Build 8D state
        state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos, gripper_qpos])

        # Images: rotate 180 degrees for LIBERO
        base_img = raw_obs["robot0_agentview_left_image"][::-1, ::-1].copy()
        wrist_img = raw_obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()

        return {
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
            "observation/state": state.astype(np.float32),
            "prompt": task_description,
        }

    def libero_actions_to_robocasa(self, actions_7d: np.ndarray) -> np.ndarray:
        """Convert LIBERO 7D actions to RoboCasa 12D PandaOmron format."""
        output_shape = actions_7d.shape[:-1] + (12,)
        actions_12d = np.zeros(output_shape, dtype=np.float32)
        actions_12d[..., :7] = actions_7d
        return actions_12d

    def predict_actions(self, obs: dict) -> np.ndarray:
        """Predict actions from RoboCasa observations."""
        raw_obs = obs.get("raw_obs", None)
        task_description = obs["task_descriptions"][0]

        if raw_obs is None:
            raw_obs = {
                "robot0_agentview_left_image": obs["main_images"][0].numpy()
                if hasattr(obs["main_images"][0], "numpy")
                else obs["main_images"][0],
                "robot0_eye_in_hand_image": obs["wrist_images"][0].numpy()
                if hasattr(obs["wrist_images"][0], "numpy")
                else obs["wrist_images"][0],
                "robot0_base_to_eef_pos": obs["states"][0, 9:12].numpy()
                if hasattr(obs["states"][0], "numpy")
                else obs["states"][0, 9:12],
                "robot0_base_to_eef_quat": obs["states"][0, 5:9].numpy()
                if hasattr(obs["states"][0], "numpy")
                else obs["states"][0, 5:9],
                "robot0_gripper_qpos": obs["states"][0, 14:16].numpy()
                if hasattr(obs["states"][0], "numpy")
                else obs["states"][0, 14:16],
            }

        # Convert to LIBERO format
        libero_obs = self.robocasa_obs_to_libero(raw_obs, task_description)

        # Use policy.infer() - handles all transforms, batching, and device placement
        result = self.policy.infer(libero_obs)
        actions_7d = result["actions"]

        # Convert to RoboCasa 12D format
        actions_12d = self.libero_actions_to_robocasa(actions_7d)

        return actions_12d[np.newaxis, ...]
