"""
Level definitions for the Gridworld.
Each level is defined as a 2D grid with entity positions.
"""
from config import GRID_WIDTH, GRID_HEIGHT


class Level:
    """Represents a single level configuration."""
    
    def __init__(self, level_id, name, grid, agent_start, description=""):
        self.level_id = level_id
        self.name = name
        self.grid = grid
        self.agent_start = agent_start
        self.description = description
        self.width = len(grid[0]) if grid else GRID_WIDTH
        self.height = len(grid) if grid else GRID_HEIGHT


# Legend: . = empty, R = rock, F = fire, A = apple, K = key, C = chest, M = monster
def parse_level(layout):
    """Parse a string-based level layout into a 2D grid."""
    char_map = {
        '.': 0,  # CELL_EMPTY
        'R': 1,  # CELL_ROCK
        'F': 2,  # CELL_FIRE
        'A': 3,  # CELL_APPLE
        'K': 4,  # CELL_KEY
        'C': 5,  # CELL_CHEST
        'M': 6,  # CELL_MONSTER
        'P': 7,  # CELL_AGENT (start position)
    }
    
    lines = [line.strip() for line in layout.strip().split('\n') if line.strip()]
    grid = []
    agent_start = (0, 0)
    
    for y, line in enumerate(lines):
        row = []
        for x, char in enumerate(line):
            if char == 'P':
                agent_start = (x, y)
                row.append(0)
            else:
                row.append(char_map.get(char, 0))
        grid.append(row)
    
    return grid, agent_start


# =============================================================================
# LEVEL 0: Basic Q-Learning - Apples Only
# =============================================================================
LEVEL_0_LAYOUT = """
P.........
..........
..........
..........
..........
..........
..........
........AA
........AA
..........
"""

# =============================================================================
# LEVEL 1: SARSA Demo - Hazards Present
# =============================================================================
LEVEL_1_LAYOUT = """
P.........
..........
..........
...FFFF...
..........
..........
..........
........AA
........AA
..........
"""

# =============================================================================
# LEVEL 2: Keys and Chests Introduction
# =============================================================================
LEVEL_2_LAYOUT = """
P.........
..........
....K.....
..........
..RRR.....
..........
..........
.......A..
........C.
..........
"""

# =============================================================================
# LEVEL 3: Complex Layout with Multiple Objectives
# =============================================================================
LEVEL_3_LAYOUT = """
P.....A...
..RR......
..........
....FF....
..........
.K........
..........
..A.......
........C.
.......A..
"""

# =============================================================================
# LEVEL 4: Monster Introduction
# =============================================================================
LEVEL_4_LAYOUT = """
P.........
..........
..........
.....M....
..........
..........
..........
........A.
........A.
..........
"""

# =============================================================================
# LEVEL 5: Multiple Monsters
# =============================================================================
LEVEL_5_LAYOUT = """
P.........
..........
...M......
..........
.......M..
..........
.K........
........C.
........A.
..........
"""

# =============================================================================
# LEVEL 6: Intrinsic Reward Testing - Sparse Rewards
# =============================================================================
LEVEL_6_LAYOUT = """
P.........
.RRRRRRRR.
..........
..........
..........
..........
..........
..........
.RRRRRRRR.
.........A
"""


def get_all_levels():
    """Return all level configurations."""
    levels = []
    
    layouts = [
        (0, "Basic Apples", LEVEL_0_LAYOUT, "Q-Learning intro - simple path to apples"),
        (1, "Fire Hazards", LEVEL_1_LAYOUT, "SARSA demo - navigate around fire"),
        (2, "Keys & Chests", LEVEL_2_LAYOUT, "Collect key to open chest"),
        (3, "Complex Layout", LEVEL_3_LAYOUT, "Multiple objectives with hazards"),
        (4, "Monster Intro", LEVEL_4_LAYOUT, "Single monster with movement"),
        (5, "Monster Challenge", LEVEL_5_LAYOUT, "Multiple monsters with objectives"),
        (6, "Sparse Rewards", LEVEL_6_LAYOUT, "Intrinsic reward exploration"),
    ]
    
    for level_id, name, layout, description in layouts:
        grid, agent_start = parse_level(layout)
        levels.append(Level(level_id, name, grid, agent_start, description))
    
    return levels


def get_level(level_id):
    """Get a specific level by ID."""
    levels = get_all_levels()
    for level in levels:
        if level.level_id == level_id:
            return level
    return levels[0]
