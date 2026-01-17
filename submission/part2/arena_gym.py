"""
Gymnasium wrapper for Arena environment.
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
    """Gym wrapper for Arena."""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, control_scheme='rotation', render_mode=None):
        super().__init__()
        self.control_scheme = control_scheme
        self.render_mode = render_mode

        # action space
        if control_scheme == 'rotation':
            self.action_space = spaces.Discrete(NUM_ACTIONS_ROTATION)
        else:
            self.action_space = spaces.Discrete(NUM_ACTIONS_DIRECTIONAL)

        # observation space: 23 features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32
        )

        self.arena = None
        self.max_steps = MAX_STEPS_PER_EPISODE

    def reset(self, seed=None, options=None):
        """Reset and return initial obs."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        if self.arena is None:
            render = self.render_mode == 'human'
            self.arena = Arena(self.control_scheme, render_mode=render)

        obs = self.arena.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        """Step and return (obs, reward, terminated, truncated, info)."""
        obs, reward, done, info = self.arena.step(action)

        # gymnasium uses terminated/truncated
        terminated = not self.arena.player.active  # died
        truncated = self.arena.steps >= self.max_steps and not terminated

        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        if self.arena is not None and self.render_mode == 'human':
            self.arena.render()

    def close(self):
        if self.arena is not None:
            self.arena.close(quit_pygame=True)
            self.arena = None


def make_arena_env(control_scheme='rotation', render_mode=None):
    """Factory function for creating arena envs."""
    return ArenaEnv(control_scheme=control_scheme, render_mode=render_mode)
