"""
Utility functions for Language-Based Exploration.

This module provides helper functions for:
- Trajectory collection with VLM integration
- Video generation and keyframe sampling
- State extraction and formatting for VLM consumption

Reuses existing components:
- SubprocessEnv from subprocess_env.py
- make_env_fn(), extract_obs() from run_rollout.py
- Pi0LiberoModel from pi0_libero_model.py
- TASK_DESCRIPTIONS from run_rollout.py
"""

import os
import io
from typing import Dict, List, Any, Optional, Callable
import numpy as np


def extract_state_from_obs(raw_obs: Dict) -> Dict[str, np.ndarray]:
    """
    Extract state information from RoboCasa observation.

    Args:
        raw_obs: Raw observation from RoboCasa environment

    Returns:
        Dictionary with eef_pos, eef_quat, gripper state
    """
    return {
        "eef_pos": raw_obs["robot0_base_to_eef_pos"].copy(),
        "eef_quat": raw_obs["robot0_base_to_eef_quat"].copy(),
        "gripper": raw_obs["robot0_gripper_qpos"].copy(),
    }


def format_state_summary(states: List[Dict[str, np.ndarray]]) -> str:
    """
    Format robot state trajectory into a text summary for VLM.

    Args:
        states: List of state dictionaries from extract_state_from_obs

    Returns:
        Text summary of the trajectory
    """
    if not states:
        return "No state information available."

    n_steps = len(states)
    lines = [f"Trajectory length: {n_steps} steps"]

    if states[0].get("eef_pos") is not None:
        start_pos = states[0]["eef_pos"]
        end_pos = states[-1]["eef_pos"]
        lines.append(f"Start EEF position: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        lines.append(f"End EEF position: [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")

        displacement = np.linalg.norm(end_pos - start_pos)
        lines.append(f"Total displacement: {displacement:.3f}m")

    if states[0].get("gripper") is not None:
        start_gripper = states[0]["gripper"]
        end_gripper = states[-1]["gripper"]
        if isinstance(start_gripper, np.ndarray):
            start_gripper = float(start_gripper.mean())
        if isinstance(end_gripper, np.ndarray):
            end_gripper = float(end_gripper.mean())
        lines.append(f"Gripper: {start_gripper:.2f} (start) -> {end_gripper:.2f} (end)")

    return "\n".join(lines)


def run_policy_with_goal(
    model,
    env,
    goal: str,
    max_steps: int = 500,
    action_chunk_size: int = 10,
    video_camera: str = "robot0_agentview_left",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single rollout with a given language goal, collecting trajectory data for VLM.

    Args:
        model: Pi0LiberoModel instance
        env: SubprocessEnv instance
        goal: Language goal/prompt for the policy
        max_steps: Maximum episode length
        action_chunk_size: Number of actions per policy query
        video_camera: Camera name for video recording
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
        - frames: List of video frames
        - states: List of state dictionaries
        - rewards: List of rewards per step
        - success: Whether task was completed
        - goal: The goal used
        - length: Episode length
        - total_reward: Cumulative reward
    """
    # Import here to avoid circular imports
    from run_rollout import extract_obs

    raw_obs = env.reset()

    frames = []
    states = []
    rewards = []
    total_reward = 0.0
    success = False
    steps = 0

    # Capture initial frame and state
    frame = raw_obs[f"{video_camera}_image"][::-1].copy()
    frames.append(frame)
    states.append(extract_state_from_obs(raw_obs))

    while steps < max_steps:
        # Get observation for policy
        obs = extract_obs(raw_obs, goal, policy_type="libero")

        # Predict action chunk
        raw_actions = model.predict_actions(obs)

        # Execute actions (LIBERO model already returns 12D RoboCasa format)
        for i in range(min(action_chunk_size, raw_actions.shape[1])):
            if steps >= max_steps:
                break

            action = raw_actions[0, i]
            raw_obs, reward, done, info = env.step(action)

            # Capture frame and state
            frame = raw_obs[f"{video_camera}_image"][::-1].copy()
            frames.append(frame)
            states.append(extract_state_from_obs(raw_obs))

            rewards.append(reward)
            total_reward += reward
            steps += 1

            if info.get("success", False):
                success = True
                if verbose:
                    print(f"  Success at step {steps}!")
                break

        if success:
            break

    if verbose:
        print(f"  Rollout complete: {steps} steps, reward={total_reward:.2f}, success={success}")

    return {
        "frames": frames,
        "states": states,
        "rewards": rewards,
        "success": success,
        "goal": goal,
        "length": steps,
        "total_reward": total_reward,
    }


def trajectory_to_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 20,
) -> None:
    """
    Save frames as MP4 video.

    Args:
        frames: List of RGB images (H, W, 3)
        output_path: Path to save video
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed. Cannot create video.")
        print("Install with: pip install imageio[ffmpeg]")
        return

    if not frames:
        return

    imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)


def trajectory_to_video_bytes(
    frames: List[np.ndarray],
    fps: int = 20,
    max_frames: int = 100,
) -> bytes:
    """
    Convert frames to MP4 video as bytes for VLM consumption.

    Args:
        frames: List of RGB images (H, W, 3)
        fps: Frames per second
        max_frames: Maximum frames to include (will subsample if exceeded)

    Returns:
        MP4 video as bytes
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio package not installed. "
            "Install with: pip install imageio[ffmpeg]"
        )

    if not frames:
        return b""

    # Subsample if too many frames
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int).tolist()
        frames = [frames[i] for i in indices]

    buffer = io.BytesIO()
    with imageio.get_writer(
        buffer,
        format="mp4",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-movflags", "frag_keyframe+empty_moov"],
    ) as writer:
        for frame in frames:
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            writer.append_data(frame)

    return buffer.getvalue()


def sample_keyframes(
    frames: List[np.ndarray],
    num_frames: int = 16,
) -> List[np.ndarray]:
    """
    Extract evenly-distributed keyframes for VLM analysis.

    Args:
        frames: Full list of trajectory frames
        num_frames: Number of frames to sample

    Returns:
        List of sampled keyframe images
    """
    if len(frames) <= num_frames:
        return frames

    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return [frames[i] for i in indices]


def format_trajectory_for_vlm(
    trajectory: Dict[str, Any],
    include_video: bool = True,
    include_keyframes: bool = False,
    include_state: bool = True,
    video_fps: int = 20,
    max_video_frames: int = 100,
    num_keyframes: int = 16,
) -> Dict[str, Any]:
    """
    Prepare trajectory data for VLM consumption.

    Args:
        trajectory: Trajectory dictionary from run_policy_with_goal
        include_video: Whether to include video bytes
        include_keyframes: Whether to include sampled keyframes
        include_state: Whether to include state summary
        video_fps: FPS for video encoding
        max_video_frames: Maximum frames in video
        num_keyframes: Number of keyframes if sampling

    Returns:
        Formatted dictionary for VLM with:
        - video: MP4 bytes (if include_video)
        - keyframes: List of images (if include_keyframes)
        - state_summary: Text summary (if include_state)
        - states: List of state dicts (if include_state)
        - goal, length, success, total_reward
    """
    result = {
        "goal": trajectory["goal"],
        "length": trajectory["length"],
        "success": trajectory["success"],
        "total_reward": trajectory.get("total_reward", 0),
    }

    if include_video and trajectory.get("frames"):
        result["video"] = trajectory_to_video_bytes(
            trajectory["frames"],
            fps=video_fps,
            max_frames=max_video_frames,
        )

    if include_keyframes and trajectory.get("frames"):
        result["keyframes"] = sample_keyframes(
            trajectory["frames"],
            num_frames=num_keyframes,
        )

    if include_state and trajectory.get("states"):
        result["states"] = trajectory["states"]
        result["state_summary"] = format_state_summary(trajectory["states"])

    return result
