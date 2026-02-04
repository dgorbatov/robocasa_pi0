"""Input/output transforms for Pi0 LIBERO - adapted from OpenPI libero_policy.py."""

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
class LiberoInputs(transforms.DataTransformFn):
    """Convert observations to Pi0 input format for LIBERO models.

    This transform handles the image and state format expected by LIBERO-finetuned models.
    LIBERO uses 8D state: [eef_pos(3), axis_angle(3), gripper_qpos(1-2)]
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) format
        base_image = parse_image(data["observation/image"])
        wrist_image = parse_image(data["observation/wrist_image"])

        # Create inputs dict matching Pi0 expected format
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad non-existent right wrist with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Only mask padding images for pi0 model, not pi0-FAST
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pass through actions during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (language instruction) to the model
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """Convert Pi0 outputs back to LIBERO action format.

    LIBERO uses 7D actions: [delta_pos(3), delta_ori(3), gripper(1)]
    """

    def __call__(self, data: dict) -> dict:
        # Return first 7 actions (LIBERO action dimension)
        return {"actions": np.asarray(data["actions"][:, :7])}
