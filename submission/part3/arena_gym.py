"""
Gym-compatible wrapper for the Arena environment.
Compatible with Stable Baselines3.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from arena import Arena
from config import (
    NUM_ACTIONS_ROTATION, NUM_ACTIONS_DIRECTIONAL,
    OBSERVATION_SIZE, MAX_STEPS_PER_EPISODE
)


class ArenaEnv(gym.Env):
    """Gymnasium wrapper for Arena environment."""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, control_scheme='rotation', render_mode=None):
        super().__init__()

        self.control_scheme = control_scheme
        self.render_mode = render_mode

        # Action space
        if control_scheme == 'rotation':
            self.action_space = spaces.Discrete(NUM_ACTIONS_ROTATION)
        else:
            self.action_space = spaces.Discrete(NUM_ACTIONS_DIRECTIONAL)

        # Observation space: 23 features for both schemes (Default_4k style)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32
        )

        # Create underlying arena (defer pygame init until first reset)
        self.arena = None
        self.max_steps = MAX_STEPS_PER_EPISODE

    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        if self.arena is None:
            render = self.render_mode == 'human'
            self.arena = Arena(self.control_scheme, render_mode=render)

        obs = self.arena.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        obs, reward, done, info = self.arena.step(action)

        # Gymnasium API uses terminated/truncated
        terminated = not self.arena.player.active  # Player died
        truncated = self.arena.steps >= self.max_steps and not terminated  # Time limit

        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.arena is not None and self.render_mode == 'human':
            self.arena.render()

    def close(self):
        """Clean up resources."""
        if self.arena is not None:
            self.arena.close(quit_pygame=True)
            self.arena = None


def make_arena_env(control_scheme='rotation', render_mode=None):
    """Factory function for creating arena environments."""
    return ArenaEnv(control_scheme=control_scheme, render_mode=render_mode)
