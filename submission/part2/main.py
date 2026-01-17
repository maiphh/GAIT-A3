"""
Main entry point with Pygame-based menu.
Deep RL Arena - Space Shooter
"""
import pygame
import sys
import os
import math
from arena import Arena
from trainer import Trainer
from config import COLORS, TRAINING


class Menu:
    """Pygame-based menu for control scheme and mode selection."""

    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Deep RL Arena - Space Shooter")
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()

        self.control_schemes = ['Rotation', 'Directional']
        self.selected_scheme = 0
        self.trainer = None

    def draw_button(self, text, x, y, width, height, selected=False):
        """Draw a menu button."""
        color = COLORS['button_hover'] if selected else COLORS['button']
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, COLORS['text'], rect, 2, border_radius=8)

        text_surface = self.font_medium.render(text, True, COLORS['text'])
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
        return rect

    def run(self):
        """Main menu loop."""
        running = True

        while running:
            self.screen.fill(COLORS['background'])

            # Title
            title = self.font_large.render("Deep RL Arena", True, COLORS['text'])
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 30))

            subtitle = self.font_small.render("Space Shooter with PPO", True, COLORS['player'])
            self.screen.blit(subtitle, (self.width // 2 - subtitle.get_width() // 2, 80))

            # Control scheme selection
            scheme_label = self.font_medium.render("Control Scheme:", True, COLORS['text'])
            self.screen.blit(scheme_label, (50, 130))

            scheme_buttons = []
            for i, scheme in enumerate(self.control_schemes):
                btn = self.draw_button(scheme, 50 + i * 260, 170, 240, 50,
                                       i == self.selected_scheme)
                scheme_buttons.append((btn, i))

            # Scheme description
            if self.selected_scheme == 0:
                desc = "Thrust, Rotate Left/Right, Shoot (5 actions)"
                desc2 = "Asteroids-style physics with momentum"
            else:
                desc = "Move Up/Down/Left/Right, Shoot (6 actions)"
                desc2 = "Direct movement, gun always shoots up"
            desc_surface = self.font_small.render(desc, True, (150, 150, 150))
            desc_surface2 = self.font_small.render(desc2, True, (150, 150, 150))
            self.screen.blit(desc_surface, (50, 235))
            self.screen.blit(desc_surface2, (50, 255))

            # Action buttons - Row 1
            action_y = 300
            play_btn = self.draw_button("Play", 50, action_y, 155, 50)
            train_btn = self.draw_button("Train", 220, action_y, 155, 50)
            demo_btn = self.draw_button("Demo", 390, action_y, 155, 50)

            # Action buttons - Row 2
            action_y2 = 365
            load_btn = self.draw_button("Load Model", 50, action_y2, 240, 50)
            graphs_btn = self.draw_button("View Graphs", 310, action_y2, 240, 50)

            # Action buttons - Row 3
            action_y3 = 430
            visual_train_btn = self.draw_button("Visual Train", 50, action_y3, 240, 50)
            train_both_btn = self.draw_button("Train Both", 310, action_y3, 240, 50)

            # Instructions
            info_y = 500
            instructions = [
                "MODES:",
                "  Play - Manual control      Train - PPO training (headless)",
                "  Demo - Watch agent         Visual Train - Watch training live",
                "  View Graphs - Training curves from logs",
                "",
                "CONTROLS (Visual Train): UP/DOWN=Speed, SPACE=Pause, ESC=Stop",
                "CONTROLS (Play): Arrow keys to move, SPACE=Shoot",
                "",
                "OBJECTIVE: Destroy spawners to advance phases (1-5)",
            ]

            for i, line in enumerate(instructions):
                color = COLORS['phase_indicator'] if line.startswith("MODES") or line.startswith("CONTROLS") or line.startswith("OBJECTIVE") else COLORS['text']
                text = self.font_small.render(line, True, color)
                self.screen.blit(text, (50, info_y + i * 20))

            # Footer
            footer = "ESC: Quit | Part 2: Deep RL with Stable Baselines3"
            footer_surface = self.font_small.render(footer, True, (100, 100, 100))
            self.screen.blit(footer_surface, (50, self.height - 30))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()

                    for btn, idx in scheme_buttons:
                        if btn.collidepoint(pos):
                            self.selected_scheme = idx

                    if play_btn.collidepoint(pos):
                        self.run_play()
                    elif train_btn.collidepoint(pos):
                        self.run_training()
                    elif demo_btn.collidepoint(pos):
                        self.run_demo()
                    elif load_btn.collidepoint(pos):
                        self.run_load_model()
                    elif graphs_btn.collidepoint(pos):
                        self.show_graphs()
                    elif visual_train_btn.collidepoint(pos):
                        self.run_visual_training()
                    elif train_both_btn.collidepoint(pos):
                        self.run_train_both()

            self.clock.tick(30)

        pygame.quit()

    def run_play(self):
        """Manual play mode with keyboard controls."""
        scheme = 'rotation' if self.selected_scheme == 0 else 'directional'
        env = Arena(control_scheme=scheme, render_mode=True)
        env.reset()
        running = True

        # Key mappings
        if scheme == 'rotation':
            key_actions = {
                pygame.K_UP: 1,      # Thrust
                pygame.K_LEFT: 2,    # Rotate left
                pygame.K_RIGHT: 3,   # Rotate right
                pygame.K_SPACE: 4,   # Shoot
            }
        else:
            key_actions = {
                pygame.K_UP: 1,      # Move up
                pygame.K_DOWN: 2,    # Move down
                pygame.K_LEFT: 3,    # Move left
                pygame.K_RIGHT: 4,   # Move right
                pygame.K_SPACE: 5,   # Shoot
            }

        while running and not env.done:
            action = 0  # Default: no action

            keys = pygame.key.get_pressed()
            for key, act in key_actions.items():
                if keys[key]:
                    action = act
                    break

            env.step(action)
            env.render(f"Manual Play - {scheme.title()} | SPACE=Shoot | ESC=Exit")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            env.clock.tick(60)

        if env.done:
            if env.victory:
                self._show_victory_screen(env)
            else:
                result = "GAME OVER" if not env.player.active else "TIME UP"
                final_msg = f"{result} | Phase: {env.phase} | Score: {int(env.total_reward)} | Press any key"
                env.render(final_msg)
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                            waiting = False
                    env.clock.tick(30)

        env.close()
        self._recreate_menu()

    def _show_victory_screen(self, env):
        """Display victory screen when player completes all 5 phases."""
        from config import WINDOW_WIDTH, WINDOW_HEIGHT

        # Create victory screen with space theme
        screen = env.screen
        clock = env.clock

        # Victory colors
        gold = (255, 215, 0)
        white = (255, 255, 255)
        dark_bg = (10, 10, 20)

        # Fonts
        font_huge = pygame.font.Font(None, 96)
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 32)

        waiting = True
        frame = 0

        while waiting:
            frame += 1
            screen.fill(dark_bg)

            # Draw stars (animated twinkle)
            for x, y, brightness in env.stars:
                twinkle = brightness + int(30 * math.sin(frame * 0.1 + x * 0.01))
                twinkle = max(50, min(200, twinkle))
                pygame.draw.circle(screen, (twinkle, twinkle, twinkle), (x, y), 1)

            # Victory text with pulsing effect
            pulse = 1.0 + 0.1 * math.sin(frame * 0.1)
            victory_text = font_huge.render("VICTORY!", True, gold)
            victory_rect = victory_text.get_rect(center=(WINDOW_WIDTH // 2, 150))
            screen.blit(victory_text, victory_rect)

            # Subtitle
            subtitle = font_large.render("All 5 Phases Completed!", True, white)
            subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 230))
            screen.blit(subtitle, subtitle_rect)

            # Stats box
            stats_y = 300
            stats = [
                f"Final Score: {int(env.total_reward):,}",
                f"Enemies Destroyed: {env.enemies_destroyed}",
                f"Spawners Destroyed: {env.spawners_destroyed}",
                f"Time: {env.steps} steps",
            ]

            for i, stat in enumerate(stats):
                stat_text = font_medium.render(stat, True, white)
                stat_rect = stat_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y + i * 40))
                screen.blit(stat_text, stat_rect)

            # Press any key prompt (blinking)
            if (frame // 30) % 2 == 0:
                prompt = font_medium.render("Press any key to continue...", True, (150, 150, 150))
                prompt_rect = prompt.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80))
                screen.blit(prompt, prompt_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False

            clock.tick(60)

    def run_training(self):
        """Execute PPO training."""
        scheme = 'rotation' if self.selected_scheme == 0 else 'directional'

        print(f"\n{'='*50}")
        print(f"Starting training: {scheme} control scheme")
        print(f"This will run headless. Check TensorBoard for progress.")
        print(f"{'='*50}\n")

        # Close pygame temporarily for training
        pygame.quit()

        self.trainer = Trainer(control_scheme=scheme)
        self.trainer.train(TRAINING['total_timesteps'])

        print("\nTraining complete! Use 'Demo' to see the trained agent.")

        # Reinitialize pygame
        pygame.init()
        self._recreate_menu()

    def run_visual_training(self):
        """Run PPO training with live visualization."""
        scheme = 'rotation' if self.selected_scheme == 0 else 'directional'

        print(f"\n{'='*50}")
        print(f"Starting VISUAL training: {scheme} control scheme")
        print(f"Controls: UP/DOWN = Speed | SPACE = Pause | ESC = Stop")
        print(f"{'='*50}\n")

        self.trainer = Trainer(control_scheme=scheme)
        self.trainer.visual_train(total_timesteps=TRAINING['total_timesteps'])

        print("\nVisual training ended. Use 'Demo' to see the trained agent.")
        self._recreate_menu()

    def run_demo(self):
        """Run visual demo of trained agent."""
        scheme = 'rotation' if self.selected_scheme == 0 else 'directional'

        if self.trainer is None or self.trainer.control_scheme != scheme:
            self.trainer = Trainer(control_scheme=scheme)

        try:
            self.trainer.load_model()
            self.trainer.demo(episodes=3)
        except FileNotFoundError:
            print(f"\nNo trained model found for {scheme} control.")
            print("Please train a model first using 'Train' button.")

        self._recreate_menu()

    def run_load_model(self):
        """Load a specific model file."""
        scheme = 'rotation' if self.selected_scheme == 0 else 'directional'
        self.trainer = Trainer(control_scheme=scheme)

        try:
            self.trainer.load_model()
            print(f"\nModel loaded for {scheme} control.")
            print("Use 'Demo' to watch the agent play.")
        except FileNotFoundError:
            print(f"\nNo model found for {scheme} control.")
            print("Train a model first.")

    def run_train_both(self):
        """Train both control schemes sequentially."""
        print(f"\n{'='*50}")
        print("Training BOTH control schemes")
        print("This will take a while...")
        print(f"{'='*50}\n")

        # Close pygame temporarily
        pygame.quit()

        from trainer import train_both_schemes
        train_both_schemes(TRAINING['total_timesteps'])

        # Reinitialize pygame
        pygame.init()
        self._recreate_menu()

    def show_graphs(self):
        """Show training graphs using matplotlib."""
        from visualization import show_training_graphs
        # use path relative to this script, not cwd
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, 'logs')
        show_training_graphs(log_dir)

    def _recreate_menu(self):
        """Recreate menu display after returning from game."""
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Deep RL Arena - Space Shooter")
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)


def main():
    menu = Menu()
    menu.run()


if __name__ == "__main__":
    main()
