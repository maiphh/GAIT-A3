"""
Gridworld environment with Pygame rendering.
Core game mechanics and entity management.
"""
import pygame
import random
import copy
from config import (
    CELL_SIZE, COLORS, ACTIONS, NUM_ACTIONS, REWARDS,
    MONSTER_MOVE_PROBABILITY, MAX_STEPS_PER_EPISODE,
    CELL_EMPTY, CELL_ROCK, CELL_FIRE, CELL_APPLE, CELL_KEY, CELL_CHEST, CELL_MONSTER
)


class Entity:
    """Base class for all grid entities."""
    
    def __init__(self, x, y, entity_type):
        self.x = x
        self.y = y
        self.entity_type = entity_type
        self.active = True
    
    def get_position(self):
        return (self.x, self.y)


class Monster(Entity):
    """Monster that moves probabilistically after agent actions."""
    
    def __init__(self, x, y):
        super().__init__(x, y, CELL_MONSTER)
    
    def try_move(self, grid_width, grid_height, blocked_positions):
        if random.random() > MONSTER_MOVE_PROBABILITY:
            return False
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if (0 <= new_x < grid_width and 0 <= new_y < grid_height and
                (new_x, new_y) not in blocked_positions):
                self.x, self.y = new_x, new_y
                return True
        return False


class Gridworld:
    """Main game environment with rendering and state management."""
    
    def __init__(self, level, render_mode=True):
        self.level = level
        self.render_mode = render_mode
        self.width = level.width
        self.height = level.height
        
        if render_mode:
            if not pygame.get_init():
                pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * CELL_SIZE, self.height * CELL_SIZE + 60)
            )
            pygame.display.set_caption(f"Gridworld - {level.name}")
            self.font = pygame.font.Font(None, 24)
            self.clock = pygame.time.Clock()
        
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.agent_x, self.agent_y = self.level.agent_start
        self.has_key = False
        self.steps = 0
        self.done = False
        self.total_reward = 0
        
        self.grid = copy.deepcopy(self.level.grid)
        self.apples = set()
        self.chests = set()
        self.monsters = []
        self.rocks = set()
        self.fires = set()
        self.key_pos = None
        
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == CELL_APPLE:
                    self.apples.add((x, y))
                elif cell == CELL_CHEST:
                    self.chests.add((x, y))
                elif cell == CELL_KEY:
                    self.key_pos = (x, y)
                elif cell == CELL_MONSTER:
                    self.monsters.append(Monster(x, y))
                elif cell == CELL_ROCK:
                    self.rocks.add((x, y))
                elif cell == CELL_FIRE:
                    self.fires.add((x, y))
        
        self.initial_collectibles = len(self.apples) + len(self.chests)
        return self.get_state()
    
    def get_state(self):
        """Return current state as a hashable tuple."""
        remaining = len(self.apples) + len(self.chests)
        monster_positions = tuple(sorted((m.x, m.y) for m in self.monsters if m.active))
        return (self.agent_x, self.agent_y, self.has_key, remaining, monster_positions)
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)."""
        if self.done:
            return self.get_state(), 0, True
        
        self.steps += 1
        reward = REWARDS['step']
        
        dx, dy = ACTIONS[action]
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy
        
        # Check bounds and rocks
        if (0 <= new_x < self.width and 0 <= new_y < self.height and
            (new_x, new_y) not in self.rocks):
            self.agent_x, self.agent_y = new_x, new_y
        
        pos = (self.agent_x, self.agent_y)
        
        # Check death conditions
        if pos in self.fires or any(pos == (m.x, m.y) for m in self.monsters if m.active):
            self.done = True
            return self.get_state(), REWARDS['death'], True
        
        # Collect items
        if pos in self.apples:
            self.apples.remove(pos)
            reward += REWARDS['apple']
        
        if pos == self.key_pos:
            self.has_key = True
            self.key_pos = None
            reward += REWARDS['key']
        
        if pos in self.chests and self.has_key:
            self.chests.remove(pos)
            self.has_key = False
            reward += REWARDS['chest']
        
        # Move monsters
        blocked = self.rocks | self.fires
        for monster in self.monsters:
            if monster.active:
                monster.try_move(self.width, self.height, blocked)
                if (self.agent_x, self.agent_y) == (monster.x, monster.y):
                    self.done = True
                    return self.get_state(), REWARDS['death'], True
        
        # Check win condition
        if len(self.apples) == 0 and len(self.chests) == 0:
            self.done = True
        
        if self.steps >= MAX_STEPS_PER_EPISODE:
            self.done = True
        
        self.total_reward += reward
        return self.get_state(), reward, self.done
    
    def render(self, info_text=""):
        """Render the current state using Pygame."""
        if not self.render_mode:
            return
        
        self.screen.fill(COLORS['background'])
        
        # Draw grid
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, COLORS['grid_line'], rect, 1)
        
        # Draw entities
        self._draw_entities()
        
        # Draw agent
        agent_rect = pygame.Rect(
            self.agent_x * CELL_SIZE + 5,
            self.agent_y * CELL_SIZE + 5,
            CELL_SIZE - 10,
            CELL_SIZE - 10
        )
        pygame.draw.ellipse(self.screen, COLORS['agent'], agent_rect)
        
        # Draw key indicator if held
        if self.has_key:
            pygame.draw.circle(
                self.screen, COLORS['key'],
                (self.agent_x * CELL_SIZE + CELL_SIZE - 10,
                 self.agent_y * CELL_SIZE + 10), 6
            )
        
        # Draw info bar
        info_y = self.height * CELL_SIZE + 5
        if info_text:
            status = info_text
        else:
            status = f"Steps: {self.steps}  Reward: {self.total_reward:.1f}"
        text_surface = self.font.render(status, True, COLORS['text'])
        self.screen.blit(text_surface, (10, info_y))
        
        pygame.display.flip()
    
    def _draw_entities(self):
        """Draw all grid entities."""
        for x, y in self.rocks:
            self._draw_cell(x, y, COLORS['rock'], 'R')
        
        for x, y in self.fires:
            self._draw_cell(x, y, COLORS['fire'], 'F')
        
        for x, y in self.apples:
            self._draw_cell(x, y, COLORS['apple'], 'A')
        
        for x, y in self.chests:
            self._draw_cell(x, y, COLORS['chest'], 'C')
        
        if self.key_pos:
            self._draw_cell(self.key_pos[0], self.key_pos[1], COLORS['key'], 'K')
        
        for monster in self.monsters:
            if monster.active:
                self._draw_cell(monster.x, monster.y, COLORS['monster'], 'M')
    
    def _draw_cell(self, x, y, color, label):
        """Draw a single cell with color and label."""
        rect = pygame.Rect(
            x * CELL_SIZE + 2, y * CELL_SIZE + 2,
            CELL_SIZE - 4, CELL_SIZE - 4
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        
        text = self.font.render(label, True, COLORS['text'])
        text_rect = text.get_rect(center=(x * CELL_SIZE + CELL_SIZE // 2,
                                          y * CELL_SIZE + CELL_SIZE // 2))
        self.screen.blit(text, text_rect)
    
    def close(self, quit_pygame=False):
        """Clean up Pygame resources.
        
        Args:
            quit_pygame: If True, fully quit pygame. If False, just leave the
                        display for the menu to recreate (faster).
        """
        if self.render_mode and quit_pygame:
            pygame.quit()
    
    def handle_events(self):
        """Process Pygame events, return False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True
