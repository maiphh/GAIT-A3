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
from config import TRAINING, ENTROPY_COEF


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


class VisualTrainingCallback(BaseCallback):
    """Callback for visual training with speed control, pause, and stop."""

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
        self.episode_reward = 0
        self.best_reward = float('-inf')
        self.recent_rewards = []

        self.clock = pygame.time.Clock()

    def _on_step(self):
        # Handle pygame events (speed, pause, stop)
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

        # Handle pause
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

            # Render paused state
            total = self.locals.get('total_timesteps', 0)
            info_text = f"Ep {self.episode+1} | Step: {self.n_calls:,}/{total:,} | Speed: {self.speed} | [PAUSED - SPACE to resume]"
            self.env.arena.render(info_text)
            self.clock.tick(30)

        # Track episode reward from the info buffer
        if len(self.model.ep_info_buffer) > 0:
            latest_ep = self.model.ep_info_buffer[-1]
            if 'r' in latest_ep:
                current_count = len(self.model.ep_info_buffer)
                if current_count > len(self.recent_rewards):
                    ep_reward = latest_ep['r']
                    self.recent_rewards.append(ep_reward)
                    self.episode = current_count

                    if ep_reward > self.best_reward:
                        self.best_reward = ep_reward

                    # Print episode summary
                    print(f"Episode {self.episode}: Reward={ep_reward:.0f}", flush=True)

        # Render current state
        total = self.locals.get('total_timesteps', 0)
        progress_pct = (self.n_calls / total * 100) if total > 0 else 0
        mean_reward = sum(self.recent_rewards[-50:]) / len(self.recent_rewards[-50:]) if self.recent_rewards else 0

        info_text = (f"Ep {self.episode+1} | "
                    f"Step: {self.n_calls:,}/{total:,} ({progress_pct:.1f}%) | "
                    f"Avg: {mean_reward:.0f} | Best: {self.best_reward:.0f} | Speed: {self.speed}")
        self.env.arena.render(info_text)

        # Control frame rate
        self.clock.tick(self.speed)

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
            ent_coef=ENTROPY_COEF,
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
        print(f"Entropy coef: {ENTROPY_COEF}")
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

        # Set the environment on the model so observation space matches
        self.model.set_env(env)

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
            arena = env.envs[0].arena

            # Check for victory
            if arena.victory:
                result = "VICTORY!"
                self._show_demo_victory(arena)
            elif not arena.player.active:
                result = "GAME OVER"
            else:
                result = "TIME UP"

            print(f"Episode {ep+1}: {result} | Reward={total_reward:.2f}, Steps={steps}, "
                  f"Phase={info_dict.get('phase', 1)}, "
                  f"Enemies={info_dict.get('enemies_destroyed', 0)}, "
                  f"Spawners={info_dict.get('spawners_destroyed', 0)}")

        env.close()
        print("\nDemo complete!")

    def _show_demo_victory(self, arena):
        """Show victory screen during demo."""
        import math
        from config import WINDOW_WIDTH, WINDOW_HEIGHT

        screen = arena.screen
        gold = (255, 215, 0)
        white = (255, 255, 255)
        dark_bg = (10, 10, 20)

        font_huge = pygame.font.Font(None, 96)
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 32)

        # Show victory for 3 seconds
        for frame in range(180):
            screen.fill(dark_bg)

            # Draw stars with twinkle
            for x, y, brightness in arena.stars:
                twinkle = brightness + int(30 * math.sin(frame * 0.1 + x * 0.01))
                twinkle = max(50, min(200, twinkle))
                pygame.draw.circle(screen, (twinkle, twinkle, twinkle), (x, y), 1)

            # Victory text
            victory_text = font_huge.render("VICTORY!", True, gold)
            victory_rect = victory_text.get_rect(center=(WINDOW_WIDTH // 2, 150))
            screen.blit(victory_text, victory_rect)

            subtitle = font_large.render("All 5 Phases Completed!", True, white)
            subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 230))
            screen.blit(subtitle, subtitle_rect)

            # Stats
            stats = [
                f"Final Score: {int(arena.total_reward):,}",
                f"Enemies Destroyed: {arena.enemies_destroyed}",
                f"Spawners Destroyed: {arena.spawners_destroyed}",
                f"Time: {arena.steps} steps",
            ]
            for i, stat in enumerate(stats):
                stat_text = font_medium.render(stat, True, white)
                stat_rect = stat_text.get_rect(center=(WINDOW_WIDTH // 2, 300 + i * 40))
                screen.blit(stat_text, stat_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            pygame.time.Clock().tick(60)

    def visual_train(self, total_timesteps=None, initial_speed=30):
        """Run PPO training with live visualization.

        Shows the agent learning in real-time using a custom rendering callback.

        Args:
            total_timesteps: Total training timesteps
            initial_speed: Initial frames per second (1-120)
        """
        total_timesteps = total_timesteps or TRAINING['total_timesteps']

        print(f"Creating visual training environment...")

        # Create environment WITH rendering
        from arena_gym import ArenaEnv
        env = ArenaEnv(control_scheme=self.control_scheme, render_mode='human')

        # Create model
        policy_kwargs = {
            'net_arch': dict(pi=TRAINING['policy_network'], vf=TRAINING['policy_network'])
        }

        self.model = PPO(
            'MlpPolicy',
            env,
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
            policy_kwargs=policy_kwargs,
            verbose=0
        )

        # Create visual training callback
        visual_callback = VisualTrainingCallback(
            env=env,
            initial_speed=initial_speed,
            model_dir=self.model_dir,
            model_name=f'ppo_{self.control_scheme}'
        )

        print(f"Starting visual training: {total_timesteps:,} timesteps")
        print(f"Speed: {initial_speed} FPS | UP/DOWN to adjust | SPACE to pause | ESC to stop\n")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=visual_callback,
                progress_bar=False
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

        # Save final model
        final_path = os.path.join(self.model_dir, f'ppo_{self.control_scheme}_final')
        self.model.save(final_path)

        env.close()

        print(f"\n{'='*50}")
        stopped_early = visual_callback.stopped_early
        print(f"Visual training {'stopped early' if stopped_early else 'complete'}!")
        print(f"Episodes: {visual_callback.episode}, Timesteps: {visual_callback.n_calls:,}")
        print(f"Best reward: {visual_callback.best_reward:.0f}")
        print(f"Model saved to: {final_path}")
        print(f"{'='*50}")


def train_both_schemes(timesteps=None):
    """Train models for both control schemes."""
    timesteps = timesteps or TRAINING['total_timesteps']

    print("\n" + "="*60)
    print("Training both control schemes")
    print("="*60)

    # Train directional control FIRST (testing order dependency)
    print("\n[1/2] Training DIRECTIONAL control scheme...")
    trainer_dir = Trainer(control_scheme='directional')
    trainer_dir.train(timesteps)

    # Train rotation control SECOND
    print("\n[2/2] Training ROTATION control scheme...")
    trainer_rot = Trainer(control_scheme='rotation')
    trainer_rot.train(timesteps)

    print("\n" + "="*60)
    print("Both models trained successfully!")
    print("="*60)

    return trainer_rot, trainer_dir
