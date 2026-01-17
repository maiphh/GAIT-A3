"""
Arena environment with Pygame rendering.
Core game loop and state management.
"""
import pygame
import random
import math
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, COLORS, PHASES, REWARDS,
    MAX_STEPS_PER_EPISODE, FPS, PLAYER, ENEMY, DIAGONAL
)
from entities import Player, Enemy, Spawner, Projectile


class Arena:
    """Main game environment."""

    def __init__(self, control_scheme='rotation', render_mode=True):
        self.control_scheme = control_scheme
        self.render_mode = render_mode

        if render_mode:
            if not pygame.get_init():
                pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption(f"Deep RL Arena - {control_scheme.title()} Control")
            self.font = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 36)
            self.clock = pygame.time.Clock()

        self._generate_stars()
        self.reset()

    def _generate_stars(self):
        """Static star background."""
        random.seed(42)
        self.stars = []
        for _ in range(100):
            x = random.randint(0, WINDOW_WIDTH)
            y = random.randint(0, WINDOW_HEIGHT)
            brightness = random.randint(50, 150)
            self.stars.append((x, y, brightness))
        random.seed()

    def reset(self):
        """Reset to initial state, return observation."""
        self.player = Player()
        self.player.control_scheme = self.control_scheme

        self.enemies = []
        self.spawners = []
        self.projectiles = []

        self.phase = 1
        self.steps = 0
        self.done = False
        self.victory = False
        self.total_reward = 0
        self.enemies_destroyed = 0
        self.spawners_destroyed = 0

        self._spawn_phase_spawners()
        return self._get_obs()

    def _spawn_phase_spawners(self):
        """Spawn spawners for current phase."""
        cfg = PHASES.get(self.phase, PHASES[5])
        n_spawners = cfg['spawners']

        for i in range(n_spawners):
            attempts = 0
            while attempts < 100:
                x = random.randint(80, WINDOW_WIDTH - 80)
                y = random.randint(80, WINDOW_HEIGHT - 80)

                # distance from player
                dx = x - self.player.x
                dy = y - self.player.y
                dist_player = math.sqrt(dx*dx + dy*dy)

                # distance from other spawners
                min_dist_spawner = float('inf')
                for s in self.spawners:
                    if s.active:
                        sdx = x - s.x
                        sdy = y - s.y
                        d = math.sqrt(sdx*sdx + sdy*sdy)
                        min_dist_spawner = min(min_dist_spawner, d)

                if dist_player > 200 and min_dist_spawner > 150:
                    break
                attempts += 1

            spawner = Spawner(x, y, spawner_id=len(self.spawners))
            spawner.spawn_interval = int(spawner.spawn_interval / cfg['spawn_rate'])
            self.spawners.append(spawner)

    def step(self, action):
        """Execute action, return (obs, reward, done, info)."""
        if self.done:
            return self._get_obs(), 0, True, self._get_info()

        self.steps += 1
        reward = REWARDS['survival_tick']

        # player action
        if self.control_scheme == 'rotation':
            shoot = self.player.apply_action_rotation(action)
        else:
            shoot = self.player.apply_action_directional(action)

        # shooting
        if shoot:
            proj = self.player.shoot()
            if proj:
                self.projectiles.append(proj)

        self.player.update()

        # spawners
        cfg = PHASES.get(self.phase, PHASES[5])
        for spawner in self.spawners:
            if spawner.active:
                enemy = spawner.update(len(self.enemies))
                if enemy:
                    enemy.speed *= cfg['enemy_speed']
                    self.enemies.append(enemy)

        # enemies
        for enemy in self.enemies:
            if enemy.active:
                enemy.update(self.player)

        # projectiles
        for proj in self.projectiles:
            if proj.active:
                proj.update()

        # collisions
        reward += self._check_collisions()

        # phase progression
        active_spawners = [s for s in self.spawners if s.active]
        if len(active_spawners) == 0 and len(self.spawners) > 0:
            reward += REWARDS['phase_progress']
            self.phase += 1
            if self.phase <= 5:
                self._spawn_phase_spawners()
            else:
                # victory!
                reward += REWARDS['victory_bonus']
                self.victory = True
                self.done = True

        # cleanup
        self.enemies = [e for e in self.enemies if e.active]
        self.projectiles = [p for p in self.projectiles if p.active]

        # end conditions
        if not self.player.active:
            reward += REWARDS['death']
            self.done = True
        elif self.steps >= MAX_STEPS_PER_EPISODE:
            self.done = True

        self.total_reward += reward
        return self._get_obs(), reward, self.done, self._get_info()

    def _check_collisions(self):
        """Handle all collisions, return reward delta."""
        reward = 0

        # player projectiles vs enemies
        for proj in self.projectiles:
            if proj.active and proj.owner == 'player':
                for enemy in self.enemies:
                    if enemy.active and proj.collides_with(enemy):
                        proj.active = False
                        reward += REWARDS['hit_enemy']
                        if enemy.take_damage(proj.damage):
                            reward += REWARDS['destroy_enemy']
                            self.enemies_destroyed += 1
                        break

        # player projectiles vs spawners
        for proj in self.projectiles:
            if proj.active and proj.owner == 'player':
                for spawner in self.spawners:
                    if spawner.active and proj.collides_with(spawner):
                        proj.active = False
                        reward += REWARDS['hit_spawner']
                        if spawner.take_damage(proj.damage):
                            reward += REWARDS['destroy_spawner']
                            self.spawners_destroyed += 1
                        break

        # enemies vs player
        if self.player.invulnerable <= 0:
            for enemy in self.enemies:
                if enemy.active and self.player.collides_with(enemy):
                    enemy.active = False
                    self.player.take_damage(ENEMY['damage'])
                    self.player.invulnerable = PLAYER['invulnerability_frames']
                    reward += REWARDS['damage_taken']

        return reward

    def _get_obs(self):
        """Build 23-feature observation vector.

        Player pos/vel/angle, nearest enemy info, nearest spawner info,
        health, phase, enemy count, shoot ready, wall distances, spawner count.
        """
        obs = []

        # player pos (normalized 0-1)
        obs.append(self.player.x / WINDOW_WIDTH)
        obs.append(self.player.y / WINDOW_HEIGHT)

        # player velocity (normalized -1 to 1)
        obs.append(self.player.vx / PLAYER['max_velocity'])
        obs.append(self.player.vy / PLAYER['max_velocity'])

        # player angle
        rad = math.radians(self.player.angle)
        obs.append(math.cos(rad))
        obs.append(math.sin(rad))

        # nearest enemy
        nearest_enemy = self._find_nearest(self.enemies)
        if nearest_enemy:
            dist = self.player.distance_to(nearest_enemy) / DIAGONAL
            angle = self.player.angle_to(nearest_enemy)
            obs.append(dist)
            obs.append(math.cos(angle))
            obs.append(math.sin(angle))
            max_speed = ENEMY['speed'] * 1.5
            obs.append(nearest_enemy.speed / max_speed)
        else:
            obs.extend([1.0, 0.0, 0.0, 0.0])

        # nearest spawner
        active_spawners = [s for s in self.spawners if s.active]
        nearest_spawner = self._find_nearest(active_spawners)
        if nearest_spawner:
            dist = self.player.distance_to(nearest_spawner) / DIAGONAL
            angle = self.player.angle_to(nearest_spawner)
            obs.append(dist)
            obs.append(math.cos(angle))
            obs.append(math.sin(angle))
            # facing: 1.0 = facing spawner, -1.0 = facing away
            rel_angle = angle - math.radians(self.player.angle)
            obs.append(math.cos(rel_angle))
        else:
            obs.extend([1.0, 0.0, 0.0, 0.0])

        # game state
        obs.append(self.player.health / PLAYER['max_health'])
        obs.append(self.phase / 5.0)
        obs.append(min(len(self.enemies) / 20.0, 1.0))
        obs.append(1.0 if self.player.can_shoot() else 0.0)

        # wall distances
        obs.append(self.player.x / WINDOW_WIDTH)  # left
        obs.append((WINDOW_WIDTH - self.player.x) / WINDOW_WIDTH)  # right
        obs.append(self.player.y / WINDOW_HEIGHT)  # top
        obs.append((WINDOW_HEIGHT - self.player.y) / WINDOW_HEIGHT)  # bottom

        # spawner count
        obs.append(len(active_spawners) / 5.0)

        return obs

    def _find_nearest(self, entities):
        """Find nearest active entity."""
        nearest = None
        min_dist = float('inf')
        for e in entities:
            if e.active:
                d = self.player.distance_to(e)
                if d < min_dist:
                    min_dist = d
                    nearest = e
        return nearest

    def _get_info(self):
        return {
            'phase': self.phase,
            'enemies_destroyed': self.enemies_destroyed,
            'spawners_destroyed': self.spawners_destroyed,
            'steps': self.steps,
            'player_health': self.player.health,
            'victory': self.victory,
        }

    def render(self, info_text=""):
        """Render current state with Pygame."""
        if not self.render_mode:
            return

        self.screen.fill(COLORS['background'])

        # stars
        for x, y, brightness in self.stars:
            pygame.draw.circle(self.screen, (brightness, brightness, brightness), (x, y), 1)

        # entities
        for spawner in self.spawners:
            if spawner.active:
                spawner.draw(self.screen)

        for enemy in self.enemies:
            if enemy.active:
                enemy.draw(self.screen)

        for proj in self.projectiles:
            if proj.active:
                proj.draw(self.screen)

        if self.player.active:
            self.player.draw(self.screen)

        self._draw_hud(info_text)
        pygame.display.flip()

    def _draw_hud(self, info_text):
        """Draw heads-up display."""
        # health bar
        bar_x, bar_y = 10, 10
        bar_w, bar_h = 200, 20
        hp_ratio = max(0, self.player.health / PLAYER['max_health'])

        pygame.draw.rect(self.screen, COLORS['health_bar_bg'],
                        (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, COLORS['health_bar'],
                        (bar_x, bar_y, bar_w * hp_ratio, bar_h))
        pygame.draw.rect(self.screen, COLORS['text'],
                        (bar_x, bar_y, bar_w, bar_h), 2)

        hp_text = self.font.render(f"HP: {max(0, self.player.health)}", True, COLORS['text'])
        self.screen.blit(hp_text, (bar_x + 5, bar_y + 2))

        # phase
        phase_text = self.font_large.render(f"Phase {self.phase}", True, COLORS['phase_indicator'])
        self.screen.blit(phase_text, (WINDOW_WIDTH - 120, 10))

        # stats
        stats = self.font.render(
            f"Enemies: {self.enemies_destroyed}  |  Spawners: {self.spawners_destroyed}  |  Score: {int(self.total_reward)}",
            True, COLORS['text']
        )
        self.screen.blit(stats, (10, 40))

        # active counts
        active = sum(1 for s in self.spawners if s.active)
        counts = self.font.render(
            f"Active Spawners: {active}  |  Enemies: {len(self.enemies)}",
            True, COLORS['text']
        )
        self.screen.blit(counts, (10, 60))

        # info text
        if info_text:
            info = self.font.render(info_text, True, COLORS['text'])
            self.screen.blit(info, (10, WINDOW_HEIGHT - 30))

    def close(self, quit_pygame=False):
        if self.render_mode and quit_pygame:
            pygame.quit()

    def handle_events(self):
        """Process pygame events, return False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True
