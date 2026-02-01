#!/usr/bin/env python3
"""Minimal Pi0 RoboCasa rollout script.

Usage:
    python run_rollout.py --checkpoint /path/to/RLinf-Pi0-RoboCasa --task CloseDrawer --num_episodes 10
"""

import argparse
import os
import numpy as np
import torch
import imageio

# Set environment variables before importing robosuite
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import robosuite
import robocasa  # noqa: F401 - registers environments
from robosuite.controllers import load_composite_controller_config

from pi0_model import Pi0RobocasaModel


# Task descriptions for RoboCasa atomic tasks
TASK_DESCRIPTIONS = {
    "OpenSingleDoor": "open cabinet or microwave door",
    "CloseSingleDoor": "close cabinet or microwave door",
    "OpenDoubleDoor": "open double cabinet doors",
    "CloseDoubleDoor": "close double cabinet doors",
    "OpenDrawer": "open drawer",
    "CloseDrawer": "close drawer",
    "PnPCounterToCab": "pick and place from counter to cabinet",
    "PnPCabToCounter": "pick and place from cabinet to counter",
    "PnPCounterToSink": "pick and place from counter to sink",
    "PnPSinkToCounter": "pick and place from sink to counter",
    "PnPCounterToStove": "pick and place from counter to stove",
    "PnPStoveToCounter": "pick and place from stove to counter",
    "PnPCounterToMicrowave": "pick and place from counter to microwave",
    "PnPMicrowaveToCounter": "pick and place from microwave to counter",
    "TurnOnMicrowave": "turn on microwave",
    "TurnOffMicrowave": "turn off microwave",
    "TurnOnSinkFaucet": "turn on sink faucet",
    "TurnOffSinkFaucet": "turn off sink faucet",
    "TurnSinkSpout": "turn sink spout",
    "TurnOnStove": "turn on stove",
    "TurnOffStove": "turn off stove",
    "CoffeeSetupMug": "setup mug for coffee",
    "CoffeeServeMug": "serve coffee into mug",
    "CoffeePressButton": "press coffee machine button",
}


def make_env(task: str, seed: int = 0, video_camera: str = None):
    """Create a single RoboCasa environment.

    Args:
        task: RoboCasa task name
        seed: Random seed
        video_camera: Optional camera name for video recording (higher resolution)
    """
    controller_config = load_composite_controller_config(
        controller=None,
        robot="PandaOmron",
    )

    # Base cameras for policy input
    camera_names = ["robot0_agentview_left", "robot0_eye_in_hand"]
    camera_widths = [128, 128]
    camera_heights = [128, 128]

    # Add video camera if specified and not already in list
    # if video_camera and video_camera not in camera_names:
    #     camera_names.append(video_camera)
    #     camera_widths.append(512)
    #     camera_heights.append(512)

    env = robosuite.make(
        env_name=task,
        robots="PandaOmron",
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
    )
    return env


def extract_obs(raw_obs: dict, task_description: str) -> dict:
    """Extract and format observations for Pi0.

    Args:
        raw_obs: Raw observation dict from robosuite
        task_description: Natural language task description

    Returns:
        Formatted observation dict for Pi0 model
    """
    # Get camera images and flip vertically (OpenGL coordinate fix)
    base_img = raw_obs["robot0_agentview_left_image"][::-1].copy()
    wrist_img = raw_obs["robot0_eye_in_hand_image"][::-1].copy()

    # Build 16D state vector matching Pi0's expected format
    state = np.zeros(16, dtype=np.float32)
    state[0:2] = raw_obs["robot0_base_pos"][:2]  # base x, y
    # [2:5] zeros (padding)
    state[5:9] = raw_obs["robot0_base_to_eef_quat"]  # end-effector quaternion
    state[9:12] = raw_obs["robot0_base_to_eef_pos"]  # end-effector position
    state[12:14] = raw_obs["robot0_gripper_qvel"]  # gripper velocities
    state[14:16] = raw_obs["robot0_gripper_qpos"]  # gripper positions

    return {
        "main_images": torch.from_numpy(base_img).unsqueeze(0),
        "wrist_images": torch.from_numpy(wrist_img).unsqueeze(0),
        "states": torch.from_numpy(state).unsqueeze(0),
        "task_descriptions": [task_description],
    }


def run_episode(
    model: Pi0RobocasaModel,
    env,
    task_description: str,
    max_steps: int = 300,
    action_chunk_size: int = 10,
    verbose: bool = True,
    video_camera: str = None,
) -> dict:
    """Run a single episode.

    Args:
        model: Pi0 model wrapper
        env: RoboCasa environment
        task_description: Natural language task description
        max_steps: Maximum steps per episode
        action_chunk_size: Number of actions to execute per inference
        verbose: Print progress
        video_camera: Camera name for video recording (None to disable)

    Returns:
        Episode statistics dict with optional 'frames' key
    """
    raw_obs = env.reset()
    total_reward = 0.0
    success = False
    steps = 0
    frames = []

    # Capture initial frame if recording - render directly from simulator
    if video_camera:
        frame = env.sim.render(height=512, width=512, camera_name=video_camera)
        frames.append(frame.copy())
        print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")

    while steps < max_steps:
        # Get observation
        obs = extract_obs(raw_obs, task_description)

        # Predict action chunk
        actions = model.predict_actions(obs)  # (1, horizon, action_dim)

        # Execute actions
        for i in range(min(action_chunk_size, actions.shape[1])):
            if steps >= max_steps:
                break

            action = actions[0, i]
            raw_obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            # Capture frame for video - render directly from simulator
            if video_camera:
                frame = env.sim.render(height=512, width=512, camera_name=video_camera)
                frames.append(frame.copy())

            if info.get("success", False):
                success = True
                if verbose:
                    print(f"  Success at step {steps}!")
                break

        if success:
            break

    result = {
        "success": success,
        "total_reward": total_reward,
        "steps": steps,
    }
    if video_camera:
        result["frames"] = frames
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Pi0 RoboCasa rollouts")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../RLinf/RLinf-Pi0-RoboCasa",
        help="Path to Pi0 checkpoint directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="CloseDrawer",
        choices=list(TASK_DESCRIPTIONS.keys()),
        help="RoboCasa task name",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="video",
        help="Directory to save videos (None to disable recording)",
    )
    parser.add_argument(
        "--video_camera",
        type=str,
        default="robot0_agentview_left",
        help="Camera to use for video recording",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=20,
        help="Frames per second for video output",
    )
    args = parser.parse_args()

    print(f"Loading Pi0 model from {args.checkpoint}...")
    model = Pi0RobocasaModel(args.checkpoint, device=args.device)

    task_description = TASK_DESCRIPTIONS[args.task]
    print(f"Task: {args.task} - '{task_description}'")

    # Create video directory if needed
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
        print(f"Recording videos to {args.video_dir}")

    successes = []
    rewards = []

    for ep in range(args.num_episodes):
        print(f"\nEpisode {ep + 1}/{args.num_episodes}")
        video_camera = args.video_camera if args.video_dir else None
        env = make_env(args.task, seed=args.seed + ep, video_camera=video_camera)

        result = run_episode(
            model,
            env,
            task_description,
            max_steps=args.max_steps,
            video_camera=video_camera,
        )

        successes.append(result["success"])
        rewards.append(result["total_reward"])

        print(f"  Steps: {result['steps']}, Reward: {result['total_reward']:.2f}")

        # Save video if recording
        if args.video_dir and "frames" in result:
            status = "success" if result["success"] else "fail"
            frames = result["frames"]

            # Debug: save first and last frame as images
            imageio.imwrite(os.path.join(args.video_dir, f"{args.task}_ep{ep:03d}_frame000.png"), frames[0])
            imageio.imwrite(os.path.join(args.video_dir, f"{args.task}_ep{ep:03d}_frame{len(frames)-1:03d}.png"), frames[-1])
            print(f"  Saved debug frames: first and last of {len(frames)} frames")

            video_path = os.path.join(
                args.video_dir,
                f"{args.task}_ep{ep:03d}_{status}.mp4"
            )
            imageio.mimsave(video_path, frames, fps=args.video_fps, macro_block_size=1)
            print(f"  Saved video: {video_path}")

        env.close()

    # Print summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Task: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Success Rate: {sum(successes)}/{args.num_episodes} ({100*sum(successes)/args.num_episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


if __name__ == "__main__":
    main()
