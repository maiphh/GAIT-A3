"""
PPO training with Stable Baselines3.
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
    """Print training progress (Kaggle-friendly)."""

    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.heartbeat_freq = 500

    def _on_step(self):
        # heartbeat
        if self.n_calls % self.heartbeat_freq == 0:
            print(".", end="", flush=True)

        if self.n_calls % self.check_freq == 0:
            print(flush=True)
            total = self.locals.get('total_timesteps', 0)
            pct = (self.n_calls / total * 100) if total > 0 else 0
            msg = f"[{self.n_calls:,}/{total:,}] ({pct:.1f}%)"

            if len(self.model.ep_info_buffer) > 0:
                mean_r = sum(ep['r'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                mean_l = sum(ep['l'] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                msg += f" | Reward: {mean_r:.1f} | Length: {mean_l:.0f}"

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
        self.episode_reward = 0
        self.best_reward = float('-inf')
        self.recent_rewards = []
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

        # pause loop
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
            info = f"Ep {self.episode+1} | Step: {self.n_calls:,}/{total:,} | Speed: {self.speed} | [PAUSED]"
            self.env.arena.render(info)
            self.clock.tick(30)

        # track episodes
        if len(self.model.ep_info_buffer) > 0:
            latest = self.model.ep_info_buffer[-1]
            if 'r' in latest:
                count = len(self.model.ep_info_buffer)
                if count > len(self.recent_rewards):
                    ep_r = latest['r']
                    self.recent_rewards.append(ep_r)
                    self.episode = count

                    if ep_r > self.best_reward:
                        self.best_reward = ep_r

                    print(f"Episode {self.episode}: Reward={ep_r:.0f}", flush=True)

        # render
        total = self.locals.get('total_timesteps', 0)
        pct = (self.n_calls / total * 100) if total > 0 else 0
        mean_r = sum(self.recent_rewards[-50:]) / len(self.recent_rewards[-50:]) if self.recent_rewards else 0

        info = (f"Ep {self.episode+1} | "
                f"Step: {self.n_calls:,}/{total:,} ({pct:.1f}%) | "
                f"Avg: {mean_r:.0f} | Best: {self.best_reward:.0f} | Speed: {self.speed}")
        self.env.arena.render(info)
        self.clock.tick(self.speed)

        return True


class Trainer:
    """PPO trainer using SB3."""

    def __init__(self, control_scheme='rotation', log_dir='logs', model_dir='models'):
        self.control_scheme = control_scheme
        self.log_dir = os.path.join(log_dir, control_scheme)
        self.model_dir = os.path.join(model_dir, f'ppo_{control_scheme}')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = None
        self.env = None

    def create_env(self, render=False):
        """Create vectorized env."""
        render_mode = 'human' if render else None

        def make_env():
            return ArenaEnv(control_scheme=self.control_scheme, render_mode=render_mode)

        env = DummyVecEnv([make_env])
        env = VecMonitor(env, self.log_dir)
        return env

    def create_model(self, env):
        """Create PPO model."""
        policy_kwargs = {
            'net_arch': dict(pi=TRAINING['policy_network'], vf=TRAINING['policy_network'])
        }

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
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.log_dir,
            verbose=0
        )
        return model

    def train(self, total_timesteps=None, save_freq=50000):
        """Run training."""
        total_timesteps = total_timesteps or TRAINING['total_timesteps']

        print(f"\n{'='*50}")
        print(f"Training PPO: {self.control_scheme}")
        print(f"{'='*50}")
        print(f"Timesteps: {total_timesteps}")
        print(f"Entropy: {ENTROPY_COEF}")
        print(f"Logs: {self.log_dir}")
        print(f"Models: {self.model_dir}")
        print(f"{'='*50}\n")

        print("Creating env...", flush=True)
        self.env = self.create_env(render=False)
        print("Creating model...", flush=True)
        self.model = self.create_model(self.env)
        print("Training...", flush=True)

        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=self.model_dir,
            name_prefix=f'ppo_{self.control_scheme}'
        )
        progress_cb = ProgressCallback(check_freq=10000)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, progress_cb],
            progress_bar=False
        )

        # save final
        final_path = os.path.join(self.model_dir, f'ppo_{self.control_scheme}_final')
        self.model.save(final_path)
        print(f"\n{'='*50}")
        print(f"Done! Saved to: {final_path}")
        print(f"{'='*50}")

        self.env.close()
        return self.model

    def load_model(self, path=None):
        """Load trained model."""
        if path is None:
            path = os.path.join(self.model_dir, f'ppo_{self.control_scheme}_final')

        if not os.path.exists(path + '.zip') and not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        self.model = PPO.load(path)
        print(f"Loaded: {path}")
        return self.model

    def demo(self, episodes=5, steps_per_second=60):
        """Visual demo of trained agent."""
        if self.model is None:
            print("No model loaded. Call load_model() first.")
            return

        print(f"\nDemo: {episodes} episodes")
        print("ESC to exit\n")

        env = self.create_env(render=True)
        self.model.set_env(env)

        for ep in range(episodes):
            obs = env.reset()
            done = False
            total_r = 0
            steps = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_r += reward[0]
                steps += 1

                env.envs[0].arena.render(f"Demo {ep+1}/{episodes} | ESC to exit")

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        env.close()
                        pygame.quit()
                        return

                pygame.time.Clock().tick(steps_per_second)

            info_dict = info[0] if info else {}
            arena = env.envs[0].arena

            if arena.victory:
                result = "VICTORY!"
                self._show_victory(arena)
            elif not arena.player.active:
                result = "GAME OVER"
            else:
                result = "TIME UP"

            print(f"Episode {ep+1}: {result} | R={total_r:.0f}, Steps={steps}, "
                  f"Phase={info_dict.get('phase', 1)}")

        env.close()
        pygame.quit()
        print("\nDemo done!")

    def _show_victory(self, arena):
        """Victory screen."""
        import math
        from config import WINDOW_WIDTH, WINDOW_HEIGHT

        screen = arena.screen
        gold = (255, 215, 0)
        white = (255, 255, 255)
        dark = (10, 10, 20)

        font_huge = pygame.font.Font(None, 96)
        font_large = pygame.font.Font(None, 48)
        font_med = pygame.font.Font(None, 32)

        for frame in range(180):  # 3 seconds
            screen.fill(dark)

            # twinkle stars
            for x, y, b in arena.stars:
                twinkle = b + int(30 * math.sin(frame * 0.1 + x * 0.01))
                twinkle = max(50, min(200, twinkle))
                pygame.draw.circle(screen, (twinkle, twinkle, twinkle), (x, y), 1)

            # text
            txt = font_huge.render("VICTORY!", True, gold)
            screen.blit(txt, txt.get_rect(center=(WINDOW_WIDTH // 2, 150)))

            sub = font_large.render("All 5 Phases Completed!", True, white)
            screen.blit(sub, sub.get_rect(center=(WINDOW_WIDTH // 2, 230)))

            # stats
            stats = [
                f"Score: {int(arena.total_reward):,}",
                f"Enemies: {arena.enemies_destroyed}",
                f"Spawners: {arena.spawners_destroyed}",
                f"Steps: {arena.steps}",
            ]
            for i, s in enumerate(stats):
                t = font_med.render(s, True, white)
                screen.blit(t, t.get_rect(center=(WINDOW_WIDTH // 2, 300 + i * 40)))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return

            pygame.time.Clock().tick(60)

    def visual_train(self, total_timesteps=None, initial_speed=30):
        """Train with live visualization."""
        total_timesteps = total_timesteps or TRAINING['total_timesteps']

        print("Creating visual training env...")

        from arena_gym import ArenaEnv
        env = ArenaEnv(control_scheme=self.control_scheme, render_mode='human')

        policy_kwargs = {
            'net_arch': dict(pi=TRAINING['policy_network'], vf=TRAINING['policy_network'])
        }

        self.model = PPO(
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
            policy_kwargs=policy_kwargs,
            verbose=0
        )

        visual_cb = VisualTrainingCallback(
            env=env,
            initial_speed=initial_speed,
            model_dir=self.model_dir,
            model_name=f'ppo_{self.control_scheme}'
        )

        print(f"Visual training: {total_timesteps:,} steps")
        print(f"Speed: {initial_speed} | UP/DOWN adjust | SPACE pause | ESC stop\n")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=visual_cb,
                progress_bar=False
            )
        except KeyboardInterrupt:
            print("\nInterrupted.")

        final_path = os.path.join(self.model_dir, f'ppo_{self.control_scheme}_final')
        self.model.save(final_path)
        env.close()

        print(f"\n{'='*50}")
        print(f"{'Stopped early' if visual_cb.stopped_early else 'Complete'}!")
        print(f"Episodes: {visual_cb.episode}, Steps: {visual_cb.n_calls:,}")
        print(f"Best: {visual_cb.best_reward:.0f}")
        print(f"Saved: {final_path}")
        print(f"{'='*50}")


def train_both_schemes(timesteps=None):
    """Train both control schemes."""
    timesteps = timesteps or TRAINING['total_timesteps']

    print("\n" + "="*60)
    print("Training both schemes")
    print("="*60)

    print("\n[1/2] Directional...")
    trainer_dir = Trainer(control_scheme='directional')
    trainer_dir.train(timesteps)

    print("\n[2/2] Rotation...")
    trainer_rot = Trainer(control_scheme='rotation')
    trainer_rot.train(timesteps)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

    return trainer_rot, trainer_dir
