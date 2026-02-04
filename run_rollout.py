#!/usr/bin/env python3
"""Minimal Pi0 RoboCasa rollout script.

Usage:
    # RoboCasa policy (default):
    python run_rollout.py --checkpoint /path/to/RLinf-Pi0-RoboCasa --task CloseDrawer --num_episodes 10

    # LIBERO policy:
    python run_rollout.py --policy libero --checkpoint /path/to/RLinf-Pi05-LIBERO-SFT --task CloseDrawer --num_episodes 10
"""

import argparse
import os
import numpy as np
import torch
import imageio

from subprocess_env import SubprocessEnv


def prepare_actions_for_robocasa(raw_actions: np.ndarray) -> np.ndarray:
    """Convert Pi0 output actions to PandaOmron format.

    Pi0 outputs 12D (after RobocasaOutputs slicing), but only [5:12] contains valid data.
    Extract the valid 7D and convert to 12D PandaOmron format.

    Args:
        raw_actions: (B, H, 12) array from Pi0 model after RobocasaOutputs

    Returns:
        actions_12d: (B, H, 12) array ready for PandaOmron environment
    """
    # Extract valid 7D from positions [5:12]
    # These are: [arm_pos(3), arm_ori(3), gripper(1)]
    actions_7d = raw_actions[..., 5:12]

    # Create 12D output array
    output_shape = actions_7d.shape[:-1] + (12,)
    actions_12d = np.zeros(output_shape, dtype=np.float32)

    # Map 7D to PandaOmron's 12D format:
    # [0:7] = arm control (3 pos + 3 ori + 1 gripper)
    # [7:11] = base control (unused, zeros)
    # [11] = mode flag (0 = control arm, 1 = control base)
    actions_12d[..., 0:7] = actions_7d
    actions_12d[..., -1] = 0  # Always control arm, not base

    return actions_12d


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


def make_env_fn(task: str, seed: int = 0):
    """Create a factory function for making a RoboCasa environment.

    Returns a callable that creates the environment (for subprocess isolation).
    """
    def _make():
        # These imports happen in the subprocess
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        import robosuite
        import robocasa  # noqa: F401 - registers environments
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(
            controller=None,
            robot="PandaOmron",
        )

        camera_names = ["robot0_agentview_left", "robot0_eye_in_hand"]
        camera_widths = [128, 128]
        camera_heights = [128, 128]

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

    return _make


def make_env(task: str, seed: int = 0):
    """Create a subprocess-isolated RoboCasa environment."""
    return SubprocessEnv(make_env_fn(task, seed))


def extract_obs(raw_obs: dict, task_description: str, policy_type: str = "robocasa") -> dict:
    """Extract and format observations for Pi0.

    Args:
        raw_obs: Raw observation dict from robosuite
        task_description: Natural language task description
        policy_type: Either "robocasa" or "libero" - determines observation format

    Returns:
        Formatted observation dict for Pi0 model
    """
    # Get camera images and flip vertically (OpenGL coordinate fix)
    # Subprocess env already deep copies, but flip creates a view so we copy again
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

    obs = {
        "main_images": torch.from_numpy(base_img).unsqueeze(0),
        "wrist_images": torch.from_numpy(wrist_img).unsqueeze(0),
        "states": torch.from_numpy(state).unsqueeze(0),
        "task_descriptions": [task_description],
    }

    # For LIBERO policy, also include raw_obs for direct conversion
    if policy_type == "libero":
        obs["raw_obs"] = raw_obs

    return obs


def run_episode(
    model,
    env,
    task_description: str,
    max_steps: int = 300,
    action_chunk_size: int = 10,
    verbose: bool = True,
    video_camera: str = None,
    policy_type: str = "robocasa",
) -> dict:
    """Run a single episode.

    Args:
        model: Pi0 model wrapper (Pi0RobocasaModel or Pi0LiberoModel)
        env: RoboCasa environment
        task_description: Natural language task description
        max_steps: Maximum steps per episode
        action_chunk_size: Number of actions to execute per inference
        verbose: Print progress
        video_camera: Camera name for video recording (None to disable)
        policy_type: Either "robocasa" or "libero"

    Returns:
        Episode statistics dict with optional 'frames' key
    """
    raw_obs = env.reset()
    total_reward = 0.0
    success = False
    steps = 0
    frames = []

    # Capture initial frame if recording - subprocess already deep copies obs
    if video_camera:
        frame = raw_obs[f"{video_camera}_image"][::-1].copy()
        frames.append(frame)
        if verbose:
            print(f"  Recording from {video_camera} at {frame.shape[:2]} resolution")

    while steps < max_steps:
        # Get observation - copy images immediately before any GPU ops
        obs = extract_obs(raw_obs, task_description, policy_type=policy_type)

        # Predict action chunk
        raw_actions = model.predict_actions(obs)  # (1, horizon, action_dim)

        # For RoboCasa policy, need to convert actions; LIBERO policy returns ready-to-use actions
        if policy_type == "robocasa":
            actions = prepare_actions_for_robocasa(raw_actions)  # (1, horizon, 12)
        else:
            # LIBERO model already returns 12D RoboCasa format
            actions = raw_actions

        # Debug: print action stats on first chunk
        if steps == 0 and verbose:
            print(f"  Raw action[0,0]: {raw_actions[0, 0, :7]}...{raw_actions[0, 0, -1]}")
            print(f"  Processed action[0,0]: {actions[0, 0, :7]}...{actions[0, 0, -1]}")

        # Execute actions
        for i in range(min(action_chunk_size, actions.shape[1])):
            if steps >= max_steps:
                break

            action = actions[0, i]
            raw_obs, reward, done, info = env.step(action)

            # Capture frame for video
            if video_camera:
                frame = raw_obs[f"{video_camera}_image"][::-1].copy()
                frames.append(frame)

            total_reward += reward
            steps += 1

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
        "--policy",
        type=str,
        choices=["robocasa", "libero"],
        default="robocasa",
        help="Policy type: 'robocasa' for RoboCasa-trained model, 'libero' for LIBERO-finetuned model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./RLinf-Pi0-RoboCasa",
        help="Path to Pi0 checkpoint directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="PnPCounterToCab",
        choices=list(TASK_DESCRIPTIONS.keys()),
        help="RoboCasa task name",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
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
    parser.add_argument(
        "--pi05",
        action="store_true",
        help="Use Pi05 model configuration (for LIBERO policy)",
    )
    args = parser.parse_args()

    print(f"Loading Pi0 model from {args.checkpoint}...")
    print(f"Policy type: {args.policy}")

    if args.policy == "libero":
        from pi0_libero_model import Pi0LiberoModel
        model = Pi0LiberoModel(
            args.checkpoint,
            device=args.device,
            pi05=args.pi05,
        )
    else:
        from pi0_model import Pi0RobocasaModel
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
        env = make_env(args.task, seed=args.seed + ep)

        result = run_episode(
            model,
            env,
            task_description,
            max_steps=args.max_steps,
            video_camera=video_camera,
            policy_type=args.policy,
        )

        successes.append(result["success"])
        rewards.append(result["total_reward"])

        print(f"  Steps: {result['steps']}, Reward: {result['total_reward']:.2f}")

        # Save video if recording
        if args.video_dir and "frames" in result:
            status = "success" if result["success"] else "fail"
            frames = result["frames"]

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
