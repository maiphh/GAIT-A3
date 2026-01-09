"""
Training orchestration for Deep RL with Stable Baselines3.
"""
import os
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from arena_gym import ArenaEnv
from config import TRAINING


class ProgressCallback(BaseCallback):
    """Callback to print training progress (Kaggle-compatible)."""

    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.heartbeat_freq = 500  # Print more frequently for Kaggle visibility

    def _on_step(self):
        # Heartbeat every 500 steps to show we're alive
        if self.n_calls % self.heartbeat_freq == 0:
            print(".", end="", flush=True)

        if self.n_calls % self.check_freq == 0:
            print(flush=True)  # New line after dots
            total = self.locals.get('total_timesteps', 0)
            pct = (self.n_calls / total * 100) if total > 0 else 0
            msg = f"[{self.n_calls:,}/{total:,}] ({pct:.1f}%)"

            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum(ep['r'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                mean_length = sum(ep['l'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                msg += f" | Reward: {mean_reward:.1f} | Length: {mean_length:.0f}"

            print(msg, flush=True)
        return True


class Trainer:
    """Handles PPO training with Stable Baselines3."""

    def __init__(self, control_scheme='rotation', log_dir='logs', model_dir='models'):
        self.control_scheme = control_scheme
        self.log_dir = os.path.join(log_dir, control_scheme)
        self.model_dir = os.path.join(model_dir, f'ppo_{control_scheme}')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = None
        self.env = None

    def create_env(self, render=False):
        """Create vectorized environment."""
        render_mode = 'human' if render else None

        def make_env():
            return ArenaEnv(control_scheme=self.control_scheme, render_mode=render_mode)

        env = DummyVecEnv([make_env])
        env = VecMonitor(env, self.log_dir)
        return env

    def create_model(self, env):
        """Create PPO model with configured hyperparameters."""
        policy_kwargs = {
            'net_arch': dict(pi=TRAINING['policy_network'], vf=TRAINING['policy_network'])
        }

        model = PPO(
            'MlpPolicy',
            env,
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.log_dir,
            verbose=0
        )
        return model

    def train(self, total_timesteps=None, save_freq=50000):
        """Run training loop."""
        total_timesteps = total_timesteps or TRAINING['total_timesteps']

        print(f"\n{'='*50}")
        print(f"Starting PPO training: {self.control_scheme} control")
        print(f"{'='*50}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Logs: {self.log_dir}")
        print(f"Models: {self.model_dir}")
        print(f"{'='*50}\n")

        print("Creating environment...", flush=True)
        self.env = self.create_env(render=False)
        print("Creating PPO model...", flush=True)
        self.model = self.create_model(self.env)
        print("Starting training loop...", flush=True)

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self.model_dir,
            name_prefix=f'ppo_{self.control_scheme}'
        )

        progress_callback = ProgressCallback(check_freq=10000)

        # Train (progress_bar=False for Kaggle compatibility)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            progress_bar=False
        )

        # Save final model
        final_path = os.path.join(self.model_dir, f'ppo_{self.control_scheme}_final')
        self.model.save(final_path)
        print(f"\n{'='*50}")
        print(f"Training complete!")
        print(f"Model saved to: {final_path}")
        print(f"{'='*50}")

        self.env.close()
        return self.model

    def load_model(self, path=None):
        """Load a trained model."""
        if path is None:
            path = os.path.join(self.model_dir, f'ppo_{self.control_scheme}_final')

        if not os.path.exists(path + '.zip') and not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

        self.model = PPO.load(path)
        print(f"Loaded model from {path}")
        return self.model

    def demo(self, episodes=5, steps_per_second=60):
        """Run visual demonstration of trained agent."""
        if self.model is None:
            print("No model loaded. Call load_model() first.")
            return

        print(f"\nRunning demo: {episodes} episodes")
        print("Press ESC to exit early\n")

        env = self.create_env(render=True)

        for ep in range(episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                steps += 1

                # Render
                env.envs[0].arena.render(f"Demo Episode {ep+1}/{episodes} | ESC to exit")

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        env.close()
                        return

                pygame.time.Clock().tick(steps_per_second)

            info_dict = info[0] if info else {}
            print(f"Episode {ep+1}: Reward={total_reward:.2f}, Steps={steps}, "
                  f"Phase={info_dict.get('phase', 1)}, "
                  f"Enemies={info_dict.get('enemies_destroyed', 0)}, "
                  f"Spawners={info_dict.get('spawners_destroyed', 0)}")

        env.close()
        print("\nDemo complete!")


def train_both_schemes(timesteps=None):
    """Train models for both control schemes."""
    timesteps = timesteps or TRAINING['total_timesteps']

    print("\n" + "="*60)
    print("Training both control schemes")
    print("="*60)

    # Train rotation control
    print("\n[1/2] Training ROTATION control scheme...")
    trainer_rot = Trainer(control_scheme='rotation')
    trainer_rot.train(timesteps)

    # Train directional control
    print("\n[2/2] Training DIRECTIONAL control scheme...")
    trainer_dir = Trainer(control_scheme='directional')
    trainer_dir.train(timesteps)

    print("\n" + "="*60)
    print("Both models trained successfully!")
    print("="*60)

    return trainer_rot, trainer_dir
