"""
Curriculum Arena - extends Arena with spawner repositioning.
"""
import random
import math
from arena import Arena
from config_curriculum import CURRICULUM, WINDOW_WIDTH, WINDOW_HEIGHT


class CurriculumArena(Arena):
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
        """Move all active spawners to new random locations."""
        cfg = CURRICULUM['spawner_reposition']

        for spawner in self.spawners:
            if not spawner.active:
                continue

            attempts = 0
            while attempts < 100:
                x = random.randint(80, WINDOW_WIDTH - 80)
                y = random.randint(80, WINDOW_HEIGHT - 80)

                # check player distance
                dx = x - self.player.x
                dy = y - self.player.y
                dist_player = math.sqrt(dx*dx + dy*dy)

                # check spawner distance
                min_dist = float('inf')
                for s in self.spawners:
                    if s.active and s.spawner_id != spawner.spawner_id:
                        sdx, sdy = x - s.x, y - s.y
                        min_dist = min(min_dist, math.sqrt(sdx*sdx + sdy*sdy))

                if dist_player > cfg['min_dist_from_player'] and min_dist > cfg['min_dist_from_spawners']:
                    break
                attempts += 1

            spawner.x = x
            spawner.y = y

        self.reposition_count += 1

    def step(self, action):
        """Step with optional spawner repositioning."""
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
