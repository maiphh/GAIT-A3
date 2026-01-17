"""
Curriculum experiment config.
Imports from base config, adds curriculum settings.
"""
from config import *

# Spawner repositioning for domain randomization
CURRICULUM = {
    'spawner_reposition': {
        'interval': 500,             # check every N steps
        'probability': 0.3,          # 30% chance to reposition
        'min_dist_from_player': 200,
        'min_dist_from_spawners': 150,
    },
}
