"""
Configuration file for the Reinforcement Learning Gridworld.
All configurable parameters are centralized here.
"""

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
CELL_SIZE = 60
GRID_WIDTH = 10
GRID_HEIGHT = 10
FPS = 60
ANIMATION_SPEED = 10  # Steps per second during demo

# =============================================================================
# COLORS (RGB)
# =============================================================================
COLORS = {
    'background': (40, 44, 52),
    'grid_line': (60, 64, 72),
    'agent': (100, 180, 255),
    'apple': (255, 100, 100),
    'key': (255, 215, 0),
    'chest': (139, 90, 43),
    'chest_open': (90, 60, 30),
    'rock': (128, 128, 128),
    'fire': (255, 69, 0),
    'monster': (148, 0, 211),
    'text': (255, 255, 255),
    'button': (70, 130, 180),
    'button_hover': (100, 160, 210),
}

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
TRAINING = {
    'episodes': 5000,
    'alpha': 0.1,           # Learning rate
    'gamma': 0.99,          # Discount factor
    'epsilon_start': 1.0,   # Initial exploration rate
    'epsilon_end': 0.01,    # Final exploration rate
}

# =============================================================================
# GAME MECHANICS
# =============================================================================
REWARDS = {
    'apple': 1,
    'chest': 2,
    'key': 0.1,
    'step': -0.1,
    'death': -10,
}

MONSTER_MOVE_PROBABILITY = 0.4
MAX_STEPS_PER_EPISODE = 200

# =============================================================================
# ACTIONS
# =============================================================================
ACTIONS = {
    0: (0, -1),   # UP
    1: (0, 1),    # DOWN
    2: (-1, 0),   # LEFT
    3: (1, 0),    # RIGHT
}
ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']
NUM_ACTIONS = 4

# =============================================================================
# CELL TYPES
# =============================================================================
CELL_EMPTY = 0
CELL_ROCK = 1
CELL_FIRE = 2
CELL_APPLE = 3
CELL_KEY = 4
CELL_CHEST = 5
CELL_MONSTER = 6
CELL_AGENT = 7

# =============================================================================
# INTRINSIC REWARD
# =============================================================================
INTRINSIC_REWARD_ENABLED = False
INTRINSIC_SCALE = 1.0  # Scaling factor for intrinsic reward
