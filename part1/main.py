"""
Main entry point with Pygame-based menu.
"""
import pygame
import sys
from levels import get_all_levels, get_level
from trainer import Trainer, train_and_compare
from gridworld import Gridworld
from config import COLORS, TRAINING


class Menu:
    """Pygame-based menu for level and algorithm selection."""
    
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Gridworld - Classical Learning")
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        self.levels = get_all_levels()
        self.selected_level = 0
        self.selected_algorithm = 0
        self.algorithms = ['Q-Learning', 'SARSA']
        self.use_intrinsic = False
    
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
            title = self.font_large.render("RL Gridworld", True, COLORS['text'])
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 30))
            
            subtitle = self.font_small.render("Classical Reinforcement Learning", True, COLORS['agent'])
            self.screen.blit(subtitle, (self.width // 2 - subtitle.get_width() // 2, 80))
            
            # Level selection
            level_label = self.font_medium.render("Select Level:", True, COLORS['text'])
            self.screen.blit(level_label, (50, 130))
            
            level_buttons = []
            for i, level in enumerate(self.levels):
                btn_text = f"L{level.level_id}: {level.name}"
                btn = self.draw_button(btn_text, 50, 160 + i * 45, 500, 38, i == self.selected_level)
                level_buttons.append((btn, i))
            
            # Algorithm selection
            algo_y = 160 + len(self.levels) * 45 + 20
            algo_label = self.font_medium.render("Algorithm:", True, COLORS['text'])
            self.screen.blit(algo_label, (50, algo_y))
            
            algo_buttons = []
            for i, algo in enumerate(self.algorithms):
                btn = self.draw_button(algo, 50 + i * 260, algo_y + 40, 240, 40, i == self.selected_algorithm)
                algo_buttons.append((btn, i))
            
            # Intrinsic reward toggle
            intrinsic_y = algo_y + 90
            intrinsic_text = f"Intrinsic Reward: {'ON' if self.use_intrinsic else 'OFF'}"
            intrinsic_btn = self.draw_button(intrinsic_text, 50, intrinsic_y, 500, 40, self.use_intrinsic)
            
            # Action buttons - Row 1
            action_y = intrinsic_y + 55
            play_btn = self.draw_button("Play", 50, action_y, 120, 50)
            train_btn = self.draw_button("Train", 180, action_y, 120, 50)
            demo_btn = self.draw_button("Demo", 310, action_y, 120, 50)
            compare_btn = self.draw_button("Compare", 440, action_y, 120, 50)
            
            # Action buttons - Row 2
            action_y2 = action_y + 60
            visual_train_btn = self.draw_button("Visualize Training", 50, action_y2, 510, 50)
            
            # Info text
            info_y = action_y2 + 70
            level = self.levels[self.selected_level]
            info_text = f"Level {level.level_id}: {level.description}"
            info_surface = self.font_small.render(info_text, True, COLORS['text'])
            self.screen.blit(info_surface, (50, info_y))
            
            # Controls info
            controls = "ESC: Quit | Play: Arrow keys | Train then Demo to see learned policy"
            ctrl_surface = self.font_small.render(controls, True, (150, 150, 150))
            self.screen.blit(ctrl_surface, (50, self.height - 30))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    
                    for btn, idx in level_buttons:
                        if btn.collidepoint(pos):
                            self.selected_level = idx
                    
                    for btn, idx in algo_buttons:
                        if btn.collidepoint(pos):
                            self.selected_algorithm = idx
                    
                    if intrinsic_btn.collidepoint(pos):
                        self.use_intrinsic = not self.use_intrinsic
                    
                    if play_btn.collidepoint(pos):
                        self.run_play()
                    
                    if train_btn.collidepoint(pos):
                        self.run_training()
                    
                    if demo_btn.collidepoint(pos):
                        self.run_demo()
                    
                    if compare_btn.collidepoint(pos):
                        self.run_comparison()
                    
                    if visual_train_btn.collidepoint(pos):
                        self.run_visual_training()
            
            self.clock.tick(30)
        
        pygame.quit()
    
    def run_play(self):
        """Manual play mode with arrow key controls."""
        level = self.levels[self.selected_level]
        env = Gridworld(level, render_mode=True)
        state = env.reset()
        running = True
        
        key_to_action = {
            pygame.K_UP: 0,
            pygame.K_DOWN: 1,
            pygame.K_LEFT: 2,
            pygame.K_RIGHT: 3,
        }
        
        while running and not env.done:
            env.render("Use Arrow Keys | ESC to exit")
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in key_to_action:
                        action = key_to_action[event.key]
                        state, reward, done = env.step(action)
            
            env.clock.tick(60)
        
        if env.done:
            result = "WIN!" if len(env.apples) == 0 and len(env.chests) == 0 else "GAME OVER"
            env.render(f"{result} - Press any key")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                        waiting = False
                env.clock.tick(30)
        
        env.close()
        
        # Recreate menu display (pygame is still running)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Gridworld - Classical Learning")
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
    
    def run_training(self):
        """Execute training for selected level and algorithm."""
        level = self.levels[self.selected_level]
        algorithm = 'q_learning' if self.selected_algorithm == 0 else 'sarsa'
        
        print(f"\nStarting training: Level {level.level_id}, {algorithm.upper()}")
        print(f"Intrinsic reward: {'Enabled' if self.use_intrinsic else 'Disabled'}")
        
        self.trainer = Trainer(
            level,
            algorithm=algorithm,
            use_intrinsic=self.use_intrinsic,
            render=False
        )
        
        stats = self.trainer.train(TRAINING['episodes'])
        stats.print_summary()
        
        save_path = f"part1/results/level_{level.level_id}_{algorithm}.png"
        title = f"Level {level.level_id}: {level.name} - {algorithm.upper()}"
        if self.use_intrinsic:
            title += " (Intrinsic)"
            save_path = save_path.replace('.png', '_intrinsic.png')
        
        stats.plot_training_curves(title=title, save_path=save_path)
    
    def run_demo(self):
        """Run visual demo of trained agent."""
        if not hasattr(self, 'trainer') or self.trainer.agent is None:
            print("No trained agent. Please train first.")
            return
        
        print("\nRunning demo...")
        self.trainer.demo(steps_per_second=5)
        
        # Recreate menu display (pygame is still running)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Gridworld - Classical Learning")
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
    
    def run_comparison(self):
        """Compare Q-Learning vs SARSA on selected level."""
        level = self.levels[self.selected_level]
        train_and_compare(level, TRAINING['episodes'])
        
        # Recreate menu display (pygame is still running)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Gridworld - Classical Learning")
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
    
    def run_visual_training(self):
        """Run training with visual display showing progress."""
        level = self.levels[self.selected_level]
        algorithm = 'q_learning' if self.selected_algorithm == 0 else 'sarsa'
        
        print(f"\nStarting visual training: Level {level.level_id}, {algorithm.upper()}")
        print(f"Intrinsic reward: {'Enabled' if self.use_intrinsic else 'Disabled'}")
        print("Controls: UP/DOWN = Speed | SPACE = Pause | ESC = Stop")
        
        self.trainer = Trainer(
            level,
            algorithm=algorithm,
            use_intrinsic=self.use_intrinsic,
            render=True
        )
        
        stats = self.trainer.visual_train(TRAINING['episodes'])
        
        if len(stats.episode_rewards) > 0:
            stats.print_summary()
            
            save_path = f"results/level_{level.level_id}_{algorithm}_visual.png"
            title = f"Level {level.level_id}: {level.name} - {algorithm.upper()}"
            if self.use_intrinsic:
                title += " (Intrinsic)"
                save_path = save_path.replace('.png', '_intrinsic.png')
            
            stats.plot_training_curves(title=title, save_path=save_path)
        
        # Recreate menu display (pygame is still running)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Gridworld - Classical Learning")
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)


def main():
    menu = Menu()
    menu.run()


if __name__ == "__main__":
    main()
