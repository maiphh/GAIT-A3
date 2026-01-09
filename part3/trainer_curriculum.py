"""
Part 3 Curriculum Experiment Trainer.
Runs control vs curriculum for both control schemes.
"""
import sys
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from arena_curriculum import CurriculumArena
from config_curriculum import (
    TRAINING, CURRICULUM, NUM_ACTIONS_ROTATION, NUM_ACTIONS_DIRECTIONAL,
    OBSERVATION_SIZE, MAX_STEPS_PER_EPISODE
)


class CurriculumEnv(gym.Env):
    """Gym wrapper for CurriculumArena."""

    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, control_scheme='rotation', render_mode=None, curriculum_enabled=False):
        super().__init__()
        self.control_scheme = control_scheme
        self.render_mode = render_mode
        self.curriculum_enabled = curriculum_enabled

        if control_scheme == 'rotation':
            self.action_space = spaces.Discrete(NUM_ACTIONS_ROTATION)
        else:
            self.action_space = spaces.Discrete(NUM_ACTIONS_DIRECTIONAL)

        self.observation_space = spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), np.float32)
        self.arena = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        if self.arena is None:
            self.arena = CurriculumArena(
                self.control_scheme,
                render_mode=(self.render_mode == 'human'),
                curriculum_enabled=self.curriculum_enabled
            )

        obs = self.arena.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.arena.step(action)
        terminated = not self.arena.player.active
        truncated = self.arena.steps >= MAX_STEPS_PER_EPISODE and not terminated
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        if self.arena and self.render_mode == 'human':
            self.arena.render()

    def close(self):
        if self.arena:
            self.arena.close(quit_pygame=True)
            self.arena = None


class ProgressCallback(BaseCallback):
    """Training progress callback."""

    def __init__(self, check_freq=10000):
        super().__init__()
        self.check_freq = check_freq

    def _on_step(self):
        if self.n_calls % 500 == 0:
            print(".", end="", flush=True)
        if self.n_calls % self.check_freq == 0:
            print(flush=True)
            total = self.locals.get('total_timesteps', 0)
            pct = (self.n_calls / total * 100) if total > 0 else 0
            msg = f"[{self.n_calls:,}/{total:,}] ({pct:.1f}%)"
            if len(self.model.ep_info_buffer) > 0:
                mean_r = sum(ep['r'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                msg += f" | Reward: {mean_r:.1f}"
            print(msg, flush=True)
        return True


def train_single(control_scheme, curriculum_enabled, timesteps, log_dir='logs', model_dir='models'):
    """Train a single model."""
    exp_type = 'curriculum' if curriculum_enabled else 'control'
    name = f'{control_scheme}_{exp_type}'

    log_path = os.path.join(log_dir, name)
    model_path = os.path.join(model_dir, f'ppo_{name}')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: {control_scheme.upper()} - {exp_type.upper()}")
    print(f"Curriculum: {curriculum_enabled}")
    if curriculum_enabled:
        cfg = CURRICULUM['spawner_reposition']
        print(f"  Interval: {cfg['interval']} steps, Probability: {cfg['probability']*100:.0f}%")
    print(f"{'='*60}")

    def make_env():
        return CurriculumEnv(control_scheme, curriculum_enabled=curriculum_enabled)

    env = VecMonitor(DummyVecEnv([make_env]), log_path)

    model = PPO(
        'MlpPolicy', env,
        learning_rate=TRAINING['learning_rate'],
        n_steps=TRAINING['n_steps'],
        batch_size=TRAINING['batch_size'],
        n_epochs=TRAINING['n_epochs'],
        gamma=TRAINING['gamma'],
        gae_lambda=TRAINING['gae_lambda'],
        clip_range=TRAINING['clip_range'],
        ent_coef=TRAINING['ent_coef'],
        vf_coef=TRAINING['vf_coef'],
        max_grad_norm=TRAINING['max_grad_norm'],
        policy_kwargs={'net_arch': dict(pi=TRAINING['policy_network'], vf=TRAINING['policy_network'])},
        tensorboard_log=log_path,
        verbose=0
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[
            CheckpointCallback(50000, model_path, f'ppo_{name}'),
            ProgressCallback()
        ],
        progress_bar=False
    )

    final = os.path.join(model_path, f'ppo_{name}_final')
    model.save(final)
    print(f"\nSaved: {final}")
    env.close()
    return model


def run_full_experiment(timesteps=None):
    """
    Run full experiment: both schemes x both conditions = 4 models.
    """
    timesteps = timesteps or TRAINING['total_timesteps']

    print("\n" + "="*70)
    print("PART 3: CURRICULUM EXPERIMENT")
    print("="*70)
    print(f"Training 4 models ({timesteps:,} steps each):")
    print("  1. Rotation - Control")
    print("  2. Rotation - Curriculum")
    print("  3. Directional - Control")
    print("  4. Directional - Curriculum")
    print("="*70)

    # Rotation
    print("\n[1/4] Rotation - Control")
    train_single('rotation', False, timesteps)

    print("\n[2/4] Rotation - Curriculum")
    train_single('rotation', True, timesteps)

    # Directional
    print("\n[3/4] Directional - Control")
    train_single('directional', False, timesteps)

    print("\n[4/4] Directional - Curriculum")
    train_single('directional', True, timesteps)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nModels saved to models/")
    print("Logs saved to logs/")
    print("Compare with: tensorboard --logdir logs")


if __name__ == '__main__':
    run_full_experiment()
