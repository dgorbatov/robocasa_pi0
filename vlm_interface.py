"""
VLM Interface for Language-Based Exploration.

This module provides an abstract interface for Vision-Language Models (VLMs)
used to provide feedback on robot trajectories and generate refined goals.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class VLMInterface(ABC):
    """Abstract base class for VLM feedback models."""

    @abstractmethod
    def get_feedback(
        self,
        video: Any,
        state_summary: str,
        current_goal: str,
        task_name: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Analyze a trajectory and provide feedback on why it failed.

        Args:
            video: Video data (format depends on implementation)
            state_summary: Text summary of robot state trajectory
            current_goal: The language goal the robot was trying to achieve
            task_name: Name of the task (e.g., "CloseDrawer")
            history: List of previous attempts with their feedback

        Returns:
            Language feedback explaining what happened and why it failed
        """
        pass

    @abstractmethod
    def generate_goal(
        self,
        original_goal: str,
        history: List[Dict],
    ) -> str:
        """
        Generate a refined goal based on the history of attempts.

        Args:
            original_goal: The initial goal that was given
            history: List of attempts containing goal, feedback, reward

        Returns:
            A refined goal string that addresses previous failure modes
        """
        pass


class GeminiRoboticsER(VLMInterface):
    """
    Gemini Robotics-ER integration for trajectory analysis and goal refinement.

    Uses the google-genai SDK to interface with Gemini Robotics-ER,
    which provides specialized understanding of robot manipulation videos.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        thinking_budget: int = 1024,
        max_video_frames: int = 100,
        video_fps: int = 10,
        verbose: bool = True,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the Gemini Robotics-ER client.

        Args:
            model_name: The model identifier
            api_key: API key (if None, looks for GEMINI_API_KEY env var)
            thinking_budget: Token budget for reasoning (0-2048+)
            max_video_frames: Maximum frames to include in video
            video_fps: FPS for encoded video
            verbose: Whether to print debug information
            log_dir: Directory to save VLM inputs for debugging
        """
        import os

        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.max_video_frames = max_video_frames
        self.video_fps = video_fps
        self.verbose = verbose
        self.log_dir = log_dir
        self._call_count = 0

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            if self.verbose:
                print(f"[GeminiRoboticsER] Logging VLM inputs to: {self.log_dir}")

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key argument."
            )

        try:
            from google import genai
            from google.genai import types

            self.client = genai.Client(api_key=self.api_key)
            self.types = types
            if self.verbose:
                print(f"[GeminiRoboticsER] Initialized with model: {model_name}")
                print(f"[GeminiRoboticsER] Thinking budget: {thinking_budget}")
        except ImportError as e:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai>=0.4.0"
            ) from e

    def _save_vlm_input(
        self,
        call_type: str,
        prompt: str,
        video_bytes: Optional[bytes] = None,
        response: Optional[str] = None,
    ):
        """Save VLM input/output for debugging."""
        if not self.log_dir:
            return

        import os
        import json

        self._call_count += 1
        call_dir = os.path.join(
            self.log_dir, f"vlm_call_{self._call_count:03d}_{call_type}"
        )
        os.makedirs(call_dir, exist_ok=True)

        prompt_path = os.path.join(call_dir, "prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(prompt)

        if video_bytes:
            video_path = os.path.join(call_dir, "input_video.mp4")
            with open(video_path, "wb") as f:
                f.write(video_bytes)

        if response:
            response_path = os.path.join(call_dir, "response.txt")
            with open(response_path, "w") as f:
                f.write(response)

        metadata = {
            "call_number": self._call_count,
            "call_type": call_type,
            "model": self.model_name,
            # "thinking_budget": self.thinking_budget,
            "video_size_bytes": len(video_bytes) if video_bytes else 0,
            "prompt_length": len(prompt),
            "response_length": len(response) if response else 0,
        }
        metadata_path = os.path.join(call_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"[GeminiRoboticsER] Saved VLM inputs to: {call_dir}")

    def get_feedback(
        self,
        video: Any,
        state_summary: str,
        current_goal: str,
        task_name: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Get feedback on a trajectory from Gemini.

        Args:
            video: Video frames as numpy array, list, or bytes
            state_summary: Text summary of robot state trajectory
            current_goal: The language goal the robot was trying to achieve
            task_name: Name of the task
            history: List of previous attempts with their feedback

        Returns:
            Language feedback explaining what happened and why it failed
        """
        video_bytes = self._prepare_video(video)
        history_str = self._format_history(history or [])

        prompt = f"""You are analyzing a robot manipulation trajectory video.

TASK: {task_name}
GOAL: {current_goal}

IMPORTANT: The environment's automated success detector says this attempt FAILED.
Even if the video appears to show success, there may be subtle issues.
Trust that this is a failed attempt and analyze what went wrong.

PREVIOUS ATTEMPTS:
{history_str}

ROBOT STATE SUMMARY:
{state_summary}

Based on the video, analyze:
1. What specific actions did the robot take?
2. At what point did execution diverge from the goal?
3. What was the likely cause of failure?
4. What concrete adjustment would most likely lead to success?

Provide concise, actionable feedback (2-4 sentences)."""

        if self.verbose:
            print(f"[GeminiRoboticsER] Sending video ({len(video_bytes)} bytes) for feedback...")

        self._save_vlm_input("get_feedback", prompt, video_bytes=video_bytes)

        try:
            video_part = self.types.Part.from_bytes(
                data=video_bytes,
                mime_type="video/mp4",
            )

            # Do not use thinking_config: many models (e.g. gemini-2.0-flash) do not support it
            config = self.types.GenerateContentConfig()

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[video_part, prompt],
                config=config,
            )

            feedback = response.text.strip()
            self._save_vlm_input("get_feedback_response", prompt, response=feedback)

            if self.verbose:
                print(f"[GeminiRoboticsER] Feedback: {feedback[:200]}...")

            return feedback

        except Exception as e:
            error_msg = f"API call failed: {e}"
            if self.verbose:
                print(f"[GeminiRoboticsER] Error: {error_msg}")
            return f"[VLM Error: {error_msg}] The robot failed to complete the task."

    def generate_goal(
        self,
        original_goal: str,
        history: List[Dict],
    ) -> str:
        """
        Generate a refined goal based on feedback history.

        Args:
            original_goal: The initial goal that was given
            history: List of attempts with feedback

        Returns:
            A refined goal string that addresses previous failure modes
        """
        history_str = self._format_history(history)

        prompt = f"""ORIGINAL GOAL: {original_goal}

ATTEMPT HISTORY:
{history_str}

CRITICAL RULES:
- You must ALWAYS output a valid task instruction the robot can execute
- NEVER output meta-statements like task completion confirmations
- The environment confirmed all previous attempts FAILED
- Your output must be a direct command like "pick up X and put it in Y"

Generate a refined goal instruction that:
1. Maintains the original intent
2. Addresses the specific failure modes observed
3. Uses natural language a robot can interpret
4. Is concise but specific (1-2 sentences max)
5. Has no measurments the robot has not notion of what meters, netwons, or angles are use relative terms

Respond with ONLY the new goal, no explanation or preamble."""

        if self.verbose:
            print(f"[GeminiRoboticsER] Generating refined goal...")

        self._save_vlm_input("generate_goal", prompt)

        try:
            # Do not use thinking_config: many models (e.g. gemini-2.0-flash) do not support it
            config = self.types.GenerateContentConfig()

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=config,
            )

            new_goal = response.text.strip()
            new_goal = new_goal.strip('"\'')
            if new_goal.lower().startswith("goal:"):
                new_goal = new_goal[5:].strip()

            self._save_vlm_input("generate_goal_response", prompt, response=new_goal)

            if self.verbose:
                print(f"[GeminiRoboticsER] New goal: {new_goal}")

            return new_goal

        except Exception as e:
            error_msg = f"API call failed: {e}"
            if self.verbose:
                print(f"[GeminiRoboticsER] Error: {error_msg}")
            return f"{original_goal} (retry attempt {len(history) + 1})"

    def _prepare_video(self, video: Any) -> bytes:
        """
        Prepare video for API submission.

        Args:
            video: Video as bytes, numpy array, or list of frames

        Returns:
            MP4 video as bytes
        """
        import io

        if isinstance(video, bytes):
            if self.verbose:
                print(f"[GeminiRoboticsER] Using pre-encoded video ({len(video)} bytes)")
            return video

        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio package not installed. "
                "Install with: pip install imageio[ffmpeg]"
            )

        frames = video
        if isinstance(frames, np.ndarray):
            if frames.dtype == object or frames.ndim == 1:
                frames = [np.asarray(f) for f in frames]
            else:
                frames = [frames[i] for i in range(len(frames))]
        elif isinstance(frames, list):
            frames = [np.asarray(f) for f in frames]

        n_frames = len(frames)

        if n_frames > self.max_video_frames:
            indices = np.linspace(0, n_frames - 1, self.max_video_frames, dtype=int).tolist()
            frames = [frames[i] for i in indices]
            if self.verbose:
                print(f"[GeminiRoboticsER] Sampled {self.max_video_frames} frames from {n_frames}")

        buffer = io.BytesIO()
        with imageio.get_writer(
            buffer,
            format="mp4",
            fps=self.video_fps,
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

    def _format_history(self, history: List[Dict]) -> str:
        """Format the attempt history for the prompt."""
        if not history:
            return "No previous attempts."

        lines = []
        for i, attempt in enumerate(history):
            lines.append(f"Attempt {i + 1}:")
            lines.append(f"  Goal used: {attempt.get('goal', 'N/A')}")
            lines.append(f"  Feedback: {attempt.get('feedback', 'N/A')}")
            lines.append(f"  Episode length: {attempt.get('length', 'N/A')} steps")
            lines.append(f"  Success: {attempt.get('success', False)}")
            lines.append("")

        return "\n".join(lines)


class MockVLM(VLMInterface):
    """
    Mock VLM for testing the pipeline without API access.

    Provides rule-based feedback and goal refinement with 50 hardcoded
    refinement strategies for comprehensive testing.
    """

    def __init__(self, verbose: bool = True, shuffle_strategies: bool = False, seed: int = 0):
        """
        Initialize the mock VLM.

        Args:
            verbose: Whether to print mock responses
            shuffle_strategies: Whether to shuffle refinement strategies
            seed: Random seed for shuffling
        """
        self.verbose = verbose
        self.seed = seed

        # 50 refinement strategies organized by category
        self._refinement_strategies = [
            # Spatial approach (10 strategies)
            "approach from the left side",
            "approach from the right side",
            "approach from directly above",
            "approach from a lower angle",
            "approach from behind the object",
            "position yourself closer before attempting",
            "keep more distance initially then move in",
            "align your approach with the object's orientation",
            "come in at a 45 degree angle",
            "circle around to find a better approach angle",
            # Speed/precision (10 strategies)
            "move more slowly and deliberately",
            "use smoother, more controlled movements",
            "pause briefly before grasping",
            "reduce speed when near the target",
            "make smaller, more precise adjustments",
            "avoid jerky or sudden movements",
            "maintain steady velocity throughout",
            "decelerate gradually as you approach",
            "use gentler force when making contact",
            "be more patient with the manipulation",
            # Sequencing (10 strategies)
            "first align with the object, then proceed",
            "position the gripper before lowering",
            "establish a stable base position first",
            "complete the reach before attempting to grasp",
            "secure the grip before lifting",
            "lift straight up before moving horizontally",
            "pause after grasping to ensure stability",
            "verify alignment at each step",
            "break the task into smaller steps",
            "complete one motion before starting the next",
            # Gripper (10 strategies)
            "open the gripper wider before approaching",
            "close the gripper more firmly",
            "adjust gripper orientation to match the object",
            "ensure centered contact with the object",
            "grasp from the sides rather than top",
            "use fingertip precision for grasping",
            "apply consistent pressure when gripping",
            "reposition if the initial grasp feels unstable",
            "check gripper clearance before closing",
            "maintain grip pressure throughout the motion",
            # Attention (10 strategies)
            "pay closer attention to the object's position",
            "focus on the target location",
            "monitor the gripper-object alignment",
            "be aware of nearby obstacles",
            "track the object throughout the motion",
            "notice if the object shifts during manipulation",
            "observe the object's orientation carefully",
            "watch for signs of slippage",
            "attend to the placement accuracy",
            "verify success before releasing",
        ]

        if shuffle_strategies:
            import random
            rng = random.Random(seed)
            rng.shuffle(self._refinement_strategies)

    def get_feedback(
        self,
        video: Any,
        state_summary: str,
        current_goal: str,
        task_name: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """Generate mock feedback based on trajectory length and history."""
        history = history or []
        n_attempts = len(history)

        # Estimate trajectory length from state summary if available
        n_frames = 100  # default
        if "steps" in state_summary.lower():
            import re
            match = re.search(r"(\d+)\s*steps", state_summary.lower())
            if match:
                n_frames = int(match.group(1))

        if n_frames < 50:
            feedback = f"The robot moved briefly but stopped early after {n_frames} steps. It may have gotten stuck or lost track of the goal."
        elif n_frames < 200:
            feedback = f"The robot made progress ({n_frames} steps) but failed to complete the task. The gripper positioning seemed off."
        else:
            feedback = f"The robot attempted the full trajectory ({n_frames} steps) but couldn't achieve the goal. Consider adjusting the approach strategy."

        if n_attempts < len(self._refinement_strategies):
            feedback += f" Suggestion: {self._refinement_strategies[n_attempts]}"

        if self.verbose:
            print(f"[MockVLM] Feedback: {feedback}")

        return feedback

    def generate_goal(
        self,
        original_goal: str,
        history: List[Dict],
    ) -> str:
        """Generate a mock refined goal based on refinement strategies."""
        n_attempts = len(history)

        if n_attempts < len(self._refinement_strategies):
            strategy = self._refinement_strategies[n_attempts]

            templates = [
                f"{strategy}, then {original_goal}",
                f"{original_goal}, but {strategy}",
                f"try to {strategy} while you {original_goal}",
                f"first {strategy}, then {original_goal}",
            ]
            template_idx = n_attempts % len(templates)
            new_goal = templates[template_idx]
        else:
            new_goal = f"{original_goal} (attempt {n_attempts + 1})"

        if self.verbose:
            print(f"[MockVLM] New goal: {new_goal}")

        return new_goal


class DebugVLM(VLMInterface):
    """
    Debug VLM that prompts for manual input.

    Useful for human-in-the-loop testing and prompt engineering.
    """

    def __init__(self, save_videos: bool = True, video_dir: str = "./debug_videos"):
        """
        Initialize the debug VLM.

        Args:
            save_videos: Whether to save trajectory videos for review
            video_dir: Directory to save videos
        """
        import os
        self.save_videos = save_videos
        self.video_dir = video_dir
        self._attempt_count = 0

        if save_videos:
            os.makedirs(video_dir, exist_ok=True)

    def get_feedback(
        self,
        video: Any,
        state_summary: str,
        current_goal: str,
        task_name: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """Prompt user for feedback after showing trajectory info."""
        self._attempt_count += 1

        # Save video if provided
        if self.save_videos and video is not None:
            self._save_video(video, f"attempt_{self._attempt_count}")

        print("\n" + "=" * 60)
        print("TRAJECTORY REVIEW")
        print("=" * 60)
        print(f"Task: {task_name}")
        print(f"Goal: {current_goal}")
        print(f"\nState Summary:\n{state_summary}")

        if self.save_videos:
            print(f"\nVideo saved to: {self.video_dir}/attempt_{self._attempt_count}.mp4")

        print("\nEnter your feedback (2-4 sentences):")
        feedback = input("> ").strip()

        return feedback or "The robot did not complete the task as expected."

    def generate_goal(
        self,
        original_goal: str,
        history: List[Dict],
    ) -> str:
        """Prompt user for a refined goal."""
        print("\n" + "=" * 60)
        print("GOAL REFINEMENT")
        print("=" * 60)
        print(f"Original goal: {original_goal}")

        if history:
            print("\nAttempt history:")
            for i, attempt in enumerate(history):
                print(f"  {i + 1}. Goal: {attempt.get('goal', 'N/A')}")
                print(f"     Feedback: {attempt.get('feedback', 'N/A')}")

        print("\nEnter refined goal:")
        new_goal = input("> ").strip()

        return new_goal or original_goal

    def _save_video(self, video: Any, name: str):
        """Save video frames to file."""
        import os
        try:
            import imageio
        except ImportError:
            print("Warning: imageio not installed, cannot save video")
            return

        frames = video
        if isinstance(frames, bytes):
            # Already encoded, save directly
            path = os.path.join(self.video_dir, f"{name}.mp4")
            with open(path, "wb") as f:
                f.write(frames)
            return

        if isinstance(frames, list) and len(frames) > 0:
            path = os.path.join(self.video_dir, f"{name}.mp4")
            imageio.mimsave(path, frames, fps=20, macro_block_size=1)


class BaselineVLM(VLMInterface):
    """
    Baseline VLM that provides no language modifications.

    For control experiments to test if just retrying with the same
    goal can lead to success due to stochasticity.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the baseline VLM.

        Args:
            verbose: Whether to print messages
        """
        self.verbose = verbose

    def get_feedback(
        self,
        video: Any,
        state_summary: str,
        current_goal: str,
        task_name: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """Return generic feedback without specific guidance."""
        history = history or []
        n_attempts = len(history)
        feedback = f"Attempt {n_attempts + 1} did not succeed. Try the task again."

        if self.verbose:
            print(f"[BaselineVLM] Feedback: {feedback}")

        return feedback

    def generate_goal(
        self,
        original_goal: str,
        history: List[Dict],
    ) -> str:
        """Return the original goal unchanged."""
        if self.verbose:
            print(f"[BaselineVLM] Keeping goal unchanged: {original_goal}")

        return original_goal
