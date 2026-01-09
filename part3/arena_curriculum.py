"""
Curriculum Arena - extends part2 Arena with spawner repositioning.
"""
import sys
import os
import random
import math

from arena import Arena as BaseArena
from config_curriculum import CURRICULUM, WINDOW_WIDTH, WINDOW_HEIGHT


class CurriculumArena(BaseArena):
    """Arena with spawner repositioning for curriculum experiment."""

    def __init__(self, control_scheme='rotation', render_mode=True, curriculum_enabled=False):
        self.curriculum_enabled = curriculum_enabled
        self.reposition_count = 0
        super().__init__(control_scheme, render_mode)

    def reset(self):
        """Reset with curriculum tracking."""
        self.reposition_count = 0
        return super().reset()

    def _reposition_spawners(self):
        """Reposition all active spawners to new random locations."""
        cfg = CURRICULUM['spawner_reposition']

        for spawner in self.spawners:
            if not spawner.active:
                continue

            attempts = 0
            while attempts < 100:
                x = random.randint(80, WINDOW_WIDTH - 80)
                y = random.randint(80, WINDOW_HEIGHT - 80)

                # Check distance from player
                dx = x - self.player.x
                dy = y - self.player.y
                dist_to_player = math.sqrt(dx*dx + dy*dy)

                # Check distance from other spawners
                min_dist = float('inf')
                for s in self.spawners:
                    if s.active and s.spawner_id != spawner.spawner_id:
                        sdx, sdy = x - s.x, y - s.y
                        min_dist = min(min_dist, math.sqrt(sdx*sdx + sdy*sdy))

                if dist_to_player > cfg['min_dist_from_player'] and min_dist > cfg['min_dist_from_spawners']:
                    break
                attempts += 1

            spawner.x = x
            spawner.y = y

        self.reposition_count += 1

    def step(self, action):
        """Step with optional spawner repositioning."""
        # Check for repositioning before normal step
        if self.curriculum_enabled and not self.done:
            cfg = CURRICULUM['spawner_reposition']
            if self.steps > 0 and self.steps % cfg['interval'] == 0:
                if random.random() < cfg['probability']:
                    self._reposition_spawners()

        return super().step(action)

    def _get_info(self):
        """Add reposition count to info."""
        info = super()._get_info()
        info['reposition_count'] = self.reposition_count
        return info
