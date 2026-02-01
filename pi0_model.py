"""Minimal Pi0 model loading for RoboCasa inference."""

import glob
import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax
import safetensors.torch
import torch
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.download as download
import openpi.transforms as transforms
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import checkpoints as _checkpoints
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from robocasa_transforms import RobocasaInputs, RobocasaOutputs


@dataclass(frozen=True)
class RobocasaDataConfig(DataConfigFactory):
    """Minimal data config for RoboCasa inference."""

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig):
        data_transforms = transforms.Group(
            inputs=[RobocasaInputs(model_type=model_config.model_type)],
            outputs=[RobocasaOutputs(action_dim=12)],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return DataConfig(
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            asset_id="physical-intelligence/robocasa",
        )


class Pi0RobocasaModel:
    """Wrapper for Pi0 model with RoboCasa-specific transforms."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        action_horizon: int = 10,
        num_denoise_steps: int = 5,
    ):
        self.device = device
        self.action_horizon = action_horizon
        self.num_denoise_steps = num_denoise_steps

        self.model, self._input_transform, self._output_transform = self._load_model(
            checkpoint_path
        )
        self.model.to(device)
        self.model.eval()

    def _load_model(self, checkpoint_path: str):
        """Load Pi0 model with proper transforms."""
        checkpoint_dir = download.maybe_download(checkpoint_path)

        # Create model config
        model_config = pi0_config.Pi0Config(action_horizon=self.action_horizon)

        # Load model weights
        model = PI0Pytorch(model_config)
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]

        for weight_path in weight_paths:
            safetensors.torch.load_model(model, weight_path, strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

        # Load normalization stats
        data_config = RobocasaDataConfig().create(checkpoint_dir, model_config)
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

        # Build transforms
        input_transform = transforms.compose([
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=False),
            *data_config.model_transforms.inputs,
        ])

        output_transform = transforms.compose([
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=False),
            *data_config.data_transforms.outputs,
        ])

        return model, input_transform, output_transform

    def _prepare_observation(self, obs: dict) -> _model.Observation:
        """Transform environment observation to model input."""
        batch_size = obs["main_images"].shape[0]
        transformed_samples = []

        for i in range(batch_size):
            sample = {
                "observation/image": obs["main_images"][i].numpy(),
                "observation/wrist_image": obs["wrist_images"][i].numpy(),
                "observation/state": obs["states"][i].numpy(),
                "prompt": obs["task_descriptions"][i],
            }
            # Transpose images from (H,W,C) - should already be in this format
            # but handle (C,H,W) just in case
            for key in ["observation/image", "observation/wrist_image"]:
                img = sample[key]
                if len(img.shape) == 3 and img.shape[0] == 3:
                    sample[key] = np.transpose(img, (1, 2, 0))

            transformed = self._input_transform(sample)
            transformed_samples.append(transformed)

        # Batch transformed samples
        batched = jax.tree.map(
            lambda *arrs: torch.from_numpy(np.stack([np.asarray(a) for a in arrs])),
            *transformed_samples,
        )
        return _model.Observation.from_dict(batched)

    @torch.no_grad()
    def predict_actions(self, obs: dict) -> np.ndarray:
        """
        Predict actions from observations.

        Args:
            obs: Dictionary with keys:
                - main_images: (B, H, W, C) uint8 tensor
                - wrist_images: (B, H, W, C) uint8 tensor
                - states: (B, 16) float32 tensor
                - task_descriptions: list of strings

        Returns:
            actions: (B, action_horizon, action_dim) numpy array
        """
        observation = self._prepare_observation(obs)

        # Move to device
        observation = _model.Observation(
            images={k: img.to(self.device) for k, img in observation.images.items()},
            image_masks={k: m.to(self.device) for k, m in observation.image_masks.items()},
            tokenized_prompt=observation.tokenized_prompt.to(self.device),
            tokenized_prompt_mask=observation.tokenized_prompt_mask.to(self.device),
            state=observation.state.to(self.device),
        )

        # Sample actions using the base PI0Pytorch inference
        actions = self.model.sample_actions(
            self.device, observation, num_steps=self.num_denoise_steps
        )

        # Transform outputs back to environment space
        batch_size = actions.shape[0]
        output_actions = []
        for i in range(batch_size):
            out = self._output_transform({
                "actions": actions[i].cpu().numpy(),
                "state": observation.state[i].cpu().numpy(),
            })
            output_actions.append(out["actions"])

        return np.stack(output_actions)
