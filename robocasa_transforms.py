"""Input/output transforms for Pi0 RoboCasa - extracted from RLinf."""

import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def parse_image(image) -> np.ndarray:
    """Convert image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """Convert observations to Pi0 input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = parse_image(data["observation/image"])
        wrist_image = parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(wrist_image),  # Pad unused
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """Convert Pi0 outputs back to environment action format."""

    action_dim: int = 12  # PandaOmron: 7 arm + 5 base

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim]}
