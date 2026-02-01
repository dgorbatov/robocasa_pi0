"""Subprocess environment wrapper for RoboCasa.

Runs the robosuite environment in a separate process to avoid OpenGL context
corruption issues that occur when MuJoCo EGL and PyTorch CUDA share a process.
"""

import cloudpickle
import numpy as np
from multiprocessing import Process, Pipe


class CloudpickleWrapper:
    """Wrapper to serialize environment factory functions."""

    def __init__(self, data):
        self.data = data

    def __getstate__(self):
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data):
        self.data = cloudpickle.loads(data)


def _worker(parent_conn, child_conn, env_fn_wrapper):
    """Worker process that runs the environment."""
    import os
    # Ensure MuJoCo environment variables are set in subprocess
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    parent_conn.close()
    env = env_fn_wrapper.data()

    try:
        while True:
            try:
                cmd, data = child_conn.recv()
            except EOFError:
                break

            if cmd == "step":
                obs, reward, done, info = env.step(data)
                # Deep copy all observation arrays to avoid buffer issues
                obs_copy = {k: np.array(v) for k, v in obs.items()}
                child_conn.send((obs_copy, reward, done, info))

            elif cmd == "reset":
                obs = env.reset()
                # Deep copy all observation arrays
                obs_copy = {k: np.array(v) for k, v in obs.items()}
                child_conn.send(obs_copy)

            elif cmd == "close":
                env.close()
                child_conn.send(None)
                break

            elif cmd == "get_attr":
                child_conn.send(getattr(env, data, None))

    except KeyboardInterrupt:
        pass
    finally:
        child_conn.close()


class SubprocessEnv:
    """Wrapper that runs a robosuite environment in a subprocess.

    This avoids OpenGL context corruption issues between MuJoCo EGL and PyTorch CUDA.

    Usage:
        def make_env():
            import robosuite
            return robosuite.make(...)

        env = SubprocessEnv(make_env)
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        env.close()
    """

    def __init__(self, env_fn):
        """Initialize subprocess environment.

        Args:
            env_fn: Callable that creates and returns a robosuite environment.
                    Must be picklable (define inside a function or use lambda).
        """
        self.parent_conn, child_conn = Pipe()
        self.process = Process(
            target=_worker,
            args=(self.parent_conn, child_conn, CloudpickleWrapper(env_fn)),
            daemon=True
        )
        self.process.start()
        child_conn.close()
        self.closed = False

    def reset(self):
        """Reset the environment and return initial observation."""
        self.parent_conn.send(("reset", None))
        return self.parent_conn.recv()

    def step(self, action):
        """Take a step in the environment.

        Args:
            action: Action array to send to the environment.

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Convert to numpy if needed
        if hasattr(action, 'cpu'):
            action = action.cpu().numpy()
        action = np.asarray(action)

        self.parent_conn.send(("step", action))
        return self.parent_conn.recv()

    def get_attr(self, name):
        """Get an attribute from the environment."""
        self.parent_conn.send(("get_attr", name))
        return self.parent_conn.recv()

    def close(self):
        """Close the environment and terminate the subprocess."""
        if self.closed:
            return
        try:
            self.parent_conn.send(("close", None))
            self.parent_conn.recv()
        except (BrokenPipeError, EOFError):
            pass
        self.process.terminate()
        self.process.join(timeout=1)
        self.closed = True

    def __del__(self):
        self.close()
