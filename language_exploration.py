#!/usr/bin/env python3
"""
Language-Based Exploration for π₀ on RoboCasa.

This script implements iterative goal refinement using VLM feedback.

Algorithm:
1. Initialize goal g₀ (from RoboCasa task description)
2. For each iteration:
   a. Roll out π₀ with current goal g
   b. If success, break
   c. Get VLM feedback on trajectory
   d. Generate refined goal based on history
   e. Update goal
3. Return successful goal or history of attempts

Usage:
    # With MockVLM for testing:
    python language_exploration.py --task CloseDrawer --vlm mock --max_iterations 5

    # With Gemini (requires GEMINI_API_KEY):
    export GEMINI_API_KEY="your-key"
    python language_exploration.py --task CloseDrawer --vlm gemini --max_iterations 10

    # With baseline (no goal changes, just retry):
    python language_exploration.py --task CloseDrawer --vlm baseline --max_iterations 5
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np

from run_rollout import TASK_DESCRIPTIONS, make_env


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)

    # Seed torch if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For full determinism (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Seed JAX if available
    try:
        import jax
        # JAX uses explicit PRNG keys, but we set the global seed for any implicit randomness
        os.environ['JAX_SEED'] = str(seed)
    except ImportError:
        pass
from pi0_libero_model import Pi0LiberoModel
from language_exploration_utils import (
    run_policy_with_goal,
    trajectory_to_video,
    format_trajectory_for_vlm,
)
from vlm_interface import VLMInterface, MockVLM, DebugVLM, GeminiRoboticsER, BaselineVLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Language-Based Exploration for π₀ on RoboCasa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        default="CloseDrawer",
        choices=list(TASK_DESCRIPTIONS.keys()),
        help="RoboCasa task name",
    )

    # VLM configuration
    parser.add_argument(
        "--vlm",
        type=str,
        default="mock",
        choices=["mock", "debug", "gemini", "baseline"],
        help="VLM backend to use",
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model name (for --vlm gemini)",
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=1024,
        help="Thinking budget for Gemini (0-2048+)",
    )
    parser.add_argument(
        "--shuffle_strategies",
        action="store_true",
        help="Shuffle MockVLM refinement strategies",
    )

    # Exploration configuration
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum number of refinement iterations",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=10,
        help="Number of actions per policy query",
    )

    # Model configuration
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_libero",
        help="Path to Pi0/Pi05 checkpoint",
    )
    parser.add_argument(
        "--pi05",
        action="store_true",
        help="Use Pi05 model configuration",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs/language_exploration",
        help="Directory for results and videos",
    )
    parser.add_argument(
        "--no_save_videos",
        action="store_true",
        help="Disable saving trajectory videos",
    )
    parser.add_argument(
        "--video_camera",
        type=str,
        default="robot0_agentview_left",
        help="Camera for video recording",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=20,
        help="FPS for saved videos",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    return parser.parse_args()


def create_vlm(args, output_dir: str) -> VLMInterface:
    """Create VLM instance based on command line arguments."""
    if args.vlm == "mock":
        print(f"Using MockVLM (shuffle={args.shuffle_strategies}, seed={args.seed})")
        return MockVLM(
            verbose=True,
            shuffle_strategies=args.shuffle_strategies,
            seed=args.seed,
        )
    elif args.vlm == "debug":
        print("Using DebugVLM with manual input")
        video_dir = os.path.join(output_dir, "debug_videos")
        return DebugVLM(
            save_videos=not args.no_save_videos,
            video_dir=video_dir,
        )
    elif args.vlm == "baseline":
        print("Using BaselineVLM (no language changes, just retry)")
        return BaselineVLM(verbose=True)
    elif args.vlm == "gemini":
        vlm_log_dir = os.path.join(output_dir, "vlm_inputs")
        print(f"Using GeminiRoboticsER ({args.vlm_model})")
        print(f"  Thinking budget: {args.thinking_budget}")
        print(f"  VLM input logging: {vlm_log_dir}")
        return GeminiRoboticsER(
            model_name=args.vlm_model,
            thinking_budget=args.thinking_budget,
            verbose=True,
            log_dir=vlm_log_dir,
        )
    else:
        raise ValueError(f"Unknown VLM type: {args.vlm}")


def language_exploration(
    task: str,
    model: Pi0LiberoModel,
    vlm: VLMInterface,
    max_iterations: int = 10,
    max_steps: int = 500,
    action_chunk_size: int = 10,
    seed: int = 0,
    save_videos: bool = True,
    output_dir: str = "./logs",
    video_camera: str = "robot0_agentview_left",
    video_fps: int = 20,
    verbose: bool = True,
):
    """
    Run the language-based exploration algorithm.

    Args:
        task: RoboCasa task name
        model: Pi0LiberoModel instance
        vlm: VLM instance for feedback and goal generation
        max_iterations: Maximum exploration iterations
        max_steps: Maximum steps per episode
        action_chunk_size: Actions per policy query
        seed: Random seed for environment
        save_videos: Whether to save trajectory videos
        output_dir: Output directory for logs and videos
        video_camera: Camera for recording
        video_fps: FPS for saved videos
        verbose: Whether to print progress

    Returns:
        Dictionary with final_goal, history, success
    """
    initial_goal = TASK_DESCRIPTIONS[task]

    print(f"\n{'='*60}")
    print("LANGUAGE-BASED EXPLORATION")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Initial goal: {initial_goal}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*60}\n")

    if save_videos:
        os.makedirs(output_dir, exist_ok=True)

    history = []
    goal = initial_goal
    success = False
    final_iteration = 0

    for i in range(max_iterations):
        final_iteration = i + 1
        print(f"\n{'='*60}")
        print(f"ITERATION {i + 1}/{max_iterations}")
        print(f"{'='*60}")
        print(f"Current goal: '{goal}'")

        # Set seed for this iteration (ensures reproducible model inference)
        iteration_seed = seed + i
        set_seed(iteration_seed)

        # Create fresh environment for each iteration
        env = make_env(task, seed=iteration_seed)

        try:
            print(f"\nRolling out policy (seed={iteration_seed})...")
            trajectory = run_policy_with_goal(
                model=model,
                env=env,
                goal=goal,
                max_steps=max_steps,
                action_chunk_size=action_chunk_size,
                video_camera=video_camera,
                verbose=verbose,
            )

            # Save video
            if save_videos and trajectory.get("frames"):
                status = "success" if trajectory["success"] else "fail"
                video_path = os.path.join(
                    output_dir,
                    f"iteration_{i + 1:02d}_{status}.mp4"
                )
                trajectory_to_video(trajectory["frames"], video_path, fps=video_fps)
                print(f"Saved video to: {video_path}")

            # Check for success
            if trajectory["success"]:
                print(f"\n*** SUCCESS on iteration {i + 1}! ***")
                success = True
                break

            # Prepare trajectory for VLM
            vlm_input = format_trajectory_for_vlm(
                trajectory,
                include_video=True,
                include_state=True,
            )

            # Get VLM feedback
            print("\nGetting VLM feedback...")
            feedback = vlm.get_feedback(
                video=vlm_input.get("video"),
                state_summary=vlm_input.get("state_summary", ""),
                current_goal=goal,
                task_name=task,
                history=history,
            )
            print(f"Feedback: {feedback}")

            # Record attempt in history
            history.append({
                "iteration": i + 1,
                "goal": goal,
                "feedback": feedback,
                "reward": trajectory["total_reward"],
                "success": trajectory["success"],
                "length": trajectory["length"],
            })

            # Generate refined goal
            print("\nGenerating refined goal...")
            new_goal = vlm.generate_goal(initial_goal, history)
            print(f"New goal: '{new_goal}'")

            goal = new_goal

        finally:
            env.close()

    # Final summary
    print(f"\n{'='*60}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {final_iteration}")
    print(f"Success: {success}")
    print(f"Final goal: '{goal}'")

    if history:
        print("\nAttempt history:")
        for h in history:
            goal_preview = h['goal'][:50] + "..." if len(h['goal']) > 50 else h['goal']
            print(f"  Iter {h['iteration']}: goal='{goal_preview}', success={h['success']}, reward={h['reward']:.2f}")

    return {
        "task": task,
        "initial_goal": initial_goal,
        "final_goal": goal,
        "success": success,
        "total_iterations": final_iteration,
        "history": history,
    }


def main():
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir,
        f"{args.task}_{args.vlm}_seed{args.seed}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    config = vars(args)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    # Load model
    print(f"\nLoading Pi0 model from {args.checkpoint}...")
    model = Pi0LiberoModel(
        checkpoint_path=args.checkpoint,
        device=args.device,
        pi05=args.pi05,
    )

    # Create VLM
    vlm = create_vlm(args, output_dir)

    # Run exploration
    try:
        results = language_exploration(
            task=args.task,
            model=model,
            vlm=vlm,
            max_iterations=args.max_iterations,
            max_steps=args.max_steps,
            action_chunk_size=args.action_chunk_size,
            seed=args.seed,
            save_videos=not args.no_save_videos,
            output_dir=output_dir,
            video_camera=args.video_camera,
            video_fps=args.video_fps,
        )

        # Save results
        results["config"] = config
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

        return results

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return None


if __name__ == "__main__":
    main()
