"""
Part 3 Curriculum Experiment Config.
Imports everything from part2, adds curriculum settings.
"""
import sys
import os

# Import all from part2 config
from config import *

# Curriculum experiment settings
CURRICULUM = {
    'spawner_reposition': {
        'interval': 500,            # Check every N steps
        'probability': 0.3,         # 30% chance to reposition
        'min_dist_from_player': 200,
        'min_dist_from_spawners': 150,
    },
}
