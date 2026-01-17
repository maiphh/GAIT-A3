"""
Curriculum experiment trainer.
Runs control vs curriculum for both control schemes.
"""
import os
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from arena_curriculum import CurriculumArena
from config_curriculum import (
    TRAINING, CURRICULUM, NUM_ACTIONS_ROTATION, NUM_ACTIONS_DIRECTIONAL,
    OBSERVATION_SIZE, MAX_STEPS_PER_EPISODE, ENTROPY_COEF
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

        obs_size = OBSERVATION_SIZE if isinstance(OBSERVATION_SIZE, int) else OBSERVATION_SIZE.get(control_scheme, 23)
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_size,), np.float32)
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


class VisualTrainingCallback(BaseCallback):
    """Visual training with speed control."""

    def __init__(self, env, initial_speed=30, model_dir='models', model_name='ppo'):
        super().__init__()
        self.env = env
        self.speed = initial_speed
        self.min_speed = 1
        self.max_speed = 120
        self.paused = False
        self.stopped_early = False
        self.model_dir = model_dir
        self.model_name = model_name

        self.episode = 0
        self.best_reward = float('-inf')
        self.recent_rewards = []
        self.reposition_count = 0
        self.clock = pygame.time.Clock()

    def _on_step(self):
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stopped_early = True
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.stopped_early = True
                    return False
                elif event.key == pygame.K_UP:
                    self.speed = min(self.speed + 5, self.max_speed)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(self.speed - 5, self.min_speed)
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

        # pause
        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stopped_early = True
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.stopped_early = True
                        return False
                    elif event.key == pygame.K_SPACE:
                        self.paused = False

            total = self.locals.get('total_timesteps', 0)
            info = f"Ep {self.episode+1} | Step: {self.n_calls:,}/{total:,} | [PAUSED]"
            self.env.arena.render(info)
            self.clock.tick(30)

        # track episodes
        if len(self.model.ep_info_buffer) > 0:
            count = len(self.model.ep_info_buffer)
            if count > len(self.recent_rewards):
                ep_r = self.model.ep_info_buffer[-1]['r']
                self.recent_rewards.append(ep_r)
                self.episode = count

                if ep_r > self.best_reward:
                    self.best_reward = ep_r
                    self.model.save(os.path.join(self.model_dir, f'{self.model_name}_best'))

                print(f"Episode {self.episode}: Reward={ep_r:.0f}", flush=True)

        # track repositioning
        if hasattr(self.env.arena, 'reposition_count'):
            self.reposition_count = self.env.arena.reposition_count

        # render
        total = self.locals.get('total_timesteps', 0)
        pct = (self.n_calls / total * 100) if total > 0 else 0
        mean_r = sum(self.recent_rewards[-50:]) / len(self.recent_rewards[-50:]) if self.recent_rewards else 0

        repos_info = f" | Repos: {self.reposition_count}" if self.reposition_count > 0 else ""
        info = (f"Ep {self.episode+1} | Step: {self.n_calls:,}/{total:,} ({pct:.1f}%) | "
                f"Avg: {mean_r:.0f} | Best: {self.best_reward:.0f}{repos_info} | Speed: {self.speed}")
        self.env.arena.render(info)
        self.clock.tick(self.speed)

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
    print(f"Entropy: {ENTROPY_COEF}")
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
        ent_coef=ENTROPY_COEF,
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


def visual_train_single(control_scheme, curriculum_enabled, timesteps, initial_speed=30, log_dir='logs', model_dir='models'):
    """Train with live visualization."""
    exp_type = 'curriculum' if curriculum_enabled else 'control'
    name = f'{control_scheme}_{exp_type}'

    model_path = os.path.join(model_dir, f'ppo_{name}')
    os.makedirs(model_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VISUAL Training: {control_scheme.upper()} - {exp_type.upper()}")
    print(f"Entropy: {ENTROPY_COEF}")
    print(f"Curriculum: {curriculum_enabled}")
    if curriculum_enabled:
        cfg = CURRICULUM['spawner_reposition']
        print(f"  Interval: {cfg['interval']} steps, Probability: {cfg['probability']*100:.0f}%")
    print(f"Controls: UP/DOWN=Speed, SPACE=Pause, ESC=Stop")
    print(f"{'='*60}")

    env = CurriculumEnv(control_scheme, render_mode='human', curriculum_enabled=curriculum_enabled)

    model = PPO(
        'MlpPolicy', env,
        learning_rate=TRAINING['learning_rate'],
        n_steps=TRAINING['n_steps'],
        batch_size=TRAINING['batch_size'],
        n_epochs=TRAINING['n_epochs'],
        gamma=TRAINING['gamma'],
        gae_lambda=TRAINING['gae_lambda'],
        clip_range=TRAINING['clip_range'],
        ent_coef=ENTROPY_COEF,
        vf_coef=TRAINING['vf_coef'],
        max_grad_norm=TRAINING['max_grad_norm'],
        policy_kwargs={'net_arch': dict(pi=TRAINING['policy_network'], vf=TRAINING['policy_network'])},
        verbose=0
    )

    visual_cb = VisualTrainingCallback(
        env=env,
        initial_speed=initial_speed,
        model_dir=model_path,
        model_name=f'ppo_{name}'
    )

    try:
        model.learn(total_timesteps=timesteps, callback=visual_cb, progress_bar=False)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    final = os.path.join(model_path, f'ppo_{name}_final')
    model.save(final)
    env.close()

    print(f"\n{'='*50}")
    print(f"{'Stopped early' if visual_cb.stopped_early else 'Complete'}!")
    print(f"Episodes: {visual_cb.episode}, Steps: {visual_cb.n_calls:,}")
    print(f"Best: {visual_cb.best_reward:.0f}")
    print(f"Saved: {final}")
    print(f"{'='*50}")

    return model


def run_full_experiment(timesteps=None):
    """Run full experiment: both schemes x both conditions = 4 models."""
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

    print("\n[1/4] Rotation - Control")
    train_single('rotation', False, timesteps)

    print("\n[2/4] Rotation - Curriculum")
    train_single('rotation', True, timesteps)

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
