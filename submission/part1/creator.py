from config import COLORS, GRID_WIDTH, GRID_HEIGHT
import pygame
from levels import get_all_levels, get_level, save_custom_level
from gridworld import Gridworld


class LevelCreator:
    """Pygame-based level editor for creating custom levels."""
    
    ENTITY_TYPES = [
        (0, '.', 'Empty', (80, 80, 80)),
        (1, 'R', 'Rock', COLORS['rock']),
        (2, 'F', 'Fire', COLORS['fire']),
        (3, 'A', 'Apple', COLORS['apple']),
        (4, 'K', 'Key', COLORS['key']),
        (5, 'C', 'Chest', COLORS['chest']),
        (6, 'M', 'Monster', COLORS['monster']),
        (7, 'P', 'Agent', COLORS['agent']),
    ]
    
    def __init__(self):
        self.cell_size = 50
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.palette_width = 150
        self.bottom_panel = 120
        
        self.width = self.grid_width * self.cell_size + self.palette_width
        self.height = self.grid_height * self.cell_size + self.bottom_panel
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Level Creator")
        
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()
        
        # Grid state: 0 = empty by default
        self.grid = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        self.agent_start = None
        self.selected_entity = 0  # Currently selected entity type
        self.level_name = "My Level"
        self.name_active = False  # For text input
    
    def run(self):
        """Main editor loop. Returns level data if saved, None if cancelled."""
        running = True
        
        while running:
            self.screen.fill(COLORS['background'])
            
            # Draw grid
            self._draw_grid()
            
            # Draw palette
            self._draw_palette()
            
            # Draw bottom panel (name input, save/cancel)
            self._draw_bottom_panel()
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif self.name_active:
                        if event.key == pygame.K_BACKSPACE:
                            self.level_name = self.level_name[:-1]
                        elif event.key == pygame.K_RETURN:
                            self.name_active = False
                        elif len(self.level_name) < 20:
                            self.level_name += event.unicode
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    result = self._handle_click(pygame.mouse.get_pos())
                    if result is None or isinstance(result, int):
                        return result
            
            self.clock.tick(60)
        
        return None
    
    def _draw_grid(self):
        """Draw the editable grid."""
        grid_area_width = self.grid_width * self.cell_size
        grid_area_height = self.grid_height * self.cell_size
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Get cell color
                cell_val = self.grid[y][x]
                color = (60, 64, 72)  # Default empty
                label = ''
                
                for entity_id, char, name, entity_color in self.ENTITY_TYPES:
                    if cell_val == entity_id:
                        color = entity_color
                        label = char
                        break
                
                # Check if agent start position
                if self.agent_start == (x, y):
                    color = COLORS['agent']
                    label = 'P'
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLORS['grid_line'], rect, 1)
                
                # Draw label
                if label:
                    text = self.font.render(label, True, COLORS['text'])
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
    
    def _draw_palette(self):
        """Draw the entity selection palette."""
        palette_x = self.grid_width * self.cell_size + 10
        
        title = self.font.render("Entities", True, COLORS['text'])
        self.screen.blit(title, (palette_x, 10))
        
        self.palette_buttons = []
        for i, (entity_id, char, name, color) in enumerate(self.ENTITY_TYPES):
            btn_y = 40 + i * 35
            btn_rect = pygame.Rect(palette_x, btn_y, 130, 30)
            
            # Highlight selected
            if self.selected_entity == entity_id:
                pygame.draw.rect(self.screen, COLORS['button_hover'], btn_rect, border_radius=4)
            else:
                pygame.draw.rect(self.screen, color, btn_rect, border_radius=4)
            
            pygame.draw.rect(self.screen, COLORS['text'], btn_rect, 1, border_radius=4)
            
            text = self.font_small.render(f"{char} {name}", True, COLORS['text'])
            text_rect = text.get_rect(center=btn_rect.center)
            self.screen.blit(text, text_rect)
            
            self.palette_buttons.append((btn_rect, entity_id))
    
    def _draw_bottom_panel(self):
        """Draw name input and save/cancel buttons."""
        panel_y = self.grid_height * self.cell_size + 10
        
        # Name input
        name_label = self.font.render("Level Name:", True, COLORS['text'])
        self.screen.blit(name_label, (10, panel_y))
        
        self.name_rect = pygame.Rect(120, panel_y - 5, 200, 30)
        color = COLORS['button_hover'] if self.name_active else COLORS['button']
        pygame.draw.rect(self.screen, color, self.name_rect, border_radius=4)
        pygame.draw.rect(self.screen, COLORS['text'], self.name_rect, 1, border_radius=4)
        
        name_text = self.font.render(self.level_name, True, COLORS['text'])
        self.screen.blit(name_text, (self.name_rect.x + 5, self.name_rect.y + 5))
        
        # Save/Cancel buttons
        btn_y = panel_y + 45
        self.save_btn = pygame.Rect(10, btn_y, 100, 40)
        self.cancel_btn = pygame.Rect(120, btn_y, 100, 40)
        
        pygame.draw.rect(self.screen, (50, 150, 50), self.save_btn, border_radius=6)
        pygame.draw.rect(self.screen, COLORS['text'], self.save_btn, 2, border_radius=6)
        save_text = self.font.render("Save", True, COLORS['text'])
        self.screen.blit(save_text, save_text.get_rect(center=self.save_btn.center))
        
        pygame.draw.rect(self.screen, (150, 50, 50), self.cancel_btn, border_radius=6)
        pygame.draw.rect(self.screen, COLORS['text'], self.cancel_btn, 2, border_radius=6)
        cancel_text = self.font.render("Cancel", True, COLORS['text'])
        self.screen.blit(cancel_text, cancel_text.get_rect(center=self.cancel_btn.center))
        
        # Instructions
        instr = self.font_small.render("Click grid to place | ESC to cancel", True, (150, 150, 150))
        self.screen.blit(instr, (10, self.height - 25))
    
    def _handle_click(self, pos):
        """Handle mouse clicks. Returns level data if saved."""
        x, y = pos
        
        # Check grid click
        if x < self.grid_width * self.cell_size and y < self.grid_height * self.cell_size:
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            
            if self.selected_entity == 7:  # Agent start
                # Clear old agent position if exists
                self.agent_start = (grid_x, grid_y)
            else:
                self.grid[grid_y][grid_x] = self.selected_entity
                # Clear agent start if overwriting
                if self.agent_start == (grid_x, grid_y):
                    self.agent_start = None
        
        # Check palette buttons
        for btn_rect, entity_id in self.palette_buttons:
            if btn_rect.collidepoint(pos):
                self.selected_entity = entity_id
        
        # Check name input
        if hasattr(self, 'name_rect') and self.name_rect.collidepoint(pos):
            self.name_active = True
        else:
            self.name_active = False
        
        # Check save button
        if hasattr(self, 'save_btn') and self.save_btn.collidepoint(pos):
            if self._validate_level():
                return self._save_level()
        
        # Check cancel button
        if hasattr(self, 'cancel_btn') and self.cancel_btn.collidepoint(pos):
            return None
        
        return "continue"  # Special value to continue loop
    
    def _validate_level(self):
        """Check if level is valid (has agent start and at least one goal)."""
        if self.agent_start is None:
            print("Error: Please place agent start position (P)")
            return False
        
        # Check for at least one apple or chest
        has_goal = False
        for row in self.grid:
            for cell in row:
                if cell in [3, 5]:  # Apple or Chest
                    has_goal = True
                    break
        
        if not has_goal:
            print("Error: Please place at least one Apple (A) or Chest (C)")
            return False
        
        return True
    
    def _save_level(self):
        """Save the level and return success indicator."""
        level_id = save_custom_level(
            name=self.level_name,
            grid=self.grid,
            agent_start=self.agent_start,
            description="Custom level"
        )
        print(f"Level saved with ID: {level_id}")
        return level_id
