"""
Arena environment with Pygame rendering.
Core game mechanics and state management.
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
    """Main arena game environment."""

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

        # Generate static stars once
        self._generate_stars()

        self.reset()

    def _generate_stars(self):
        """Generate static star positions."""
        random.seed(42)
        self.stars = []
        for _ in range(100):
            x = random.randint(0, WINDOW_WIDTH)
            y = random.randint(0, WINDOW_HEIGHT)
            brightness = random.randint(50, 150)
            self.stars.append((x, y, brightness))
        random.seed()

    def reset(self):
        """Reset environment to initial state. Returns observation."""
        self.player = Player()
        self.player.control_scheme = self.control_scheme

        self.enemies = []
        self.spawners = []
        self.projectiles = []

        self.phase = 1
        self.steps = 0
        self.done = False
        self.total_reward = 0
        self.enemies_destroyed = 0
        self.spawners_destroyed = 0

        # Spawn initial spawners for phase 1
        self._spawn_phase_spawners()

        return self._get_observation()

    def _spawn_phase_spawners(self):
        """Spawn spawners for current phase."""
        phase_config = PHASES.get(self.phase, PHASES[5])
        num_spawners = phase_config['spawners']

        for i in range(num_spawners):
            # Place spawners away from player and other spawners
            attempts = 0
            while attempts < 100:
                x = random.randint(80, WINDOW_WIDTH - 80)
                y = random.randint(80, WINDOW_HEIGHT - 80)

                # Check distance from player
                dx = x - self.player.x
                dy = y - self.player.y
                dist_to_player = math.sqrt(dx*dx + dy*dy)

                # Check distance from other spawners
                min_dist_to_spawner = float('inf')
                for s in self.spawners:
                    if s.active:
                        sdx = x - s.x
                        sdy = y - s.y
                        dist = math.sqrt(sdx*sdx + sdy*sdy)
                        min_dist_to_spawner = min(min_dist_to_spawner, dist)

                if dist_to_player > 200 and min_dist_to_spawner > 150:
                    break
                attempts += 1

            spawner = Spawner(x, y, spawner_id=len(self.spawners))
            # Adjust spawn rate based on phase
            spawner.spawn_interval = int(spawner.spawn_interval / phase_config['spawn_rate'])
            self.spawners.append(spawner)

    def step(self, action):
        """Execute action and return (observation, reward, done, info)."""
        if self.done:
            return self._get_observation(), 0, True, self._get_info()

        self.steps += 1
        reward = REWARDS['survival_tick']

        # Apply player action
        if self.control_scheme == 'rotation':
            shoot = self.player.apply_action_rotation(action)
        else:
            shoot = self.player.apply_action_directional(action)

        # Handle shooting
        if shoot:
            projectile = self.player.shoot()
            if projectile:
                self.projectiles.append(projectile)

        # Update player physics
        self.player.update()

        # Update spawners and spawn enemies
        phase_config = PHASES.get(self.phase, PHASES[5])
        for spawner in self.spawners:
            if spawner.active:
                new_enemy = spawner.update(len(self.enemies))
                if new_enemy:
                    # Apply phase speed modifier
                    new_enemy.speed *= phase_config['enemy_speed']
                    self.enemies.append(new_enemy)

        # Update enemies
        for enemy in self.enemies:
            if enemy.active:
                enemy.update(self.player)

        # Update projectiles
        for proj in self.projectiles:
            if proj.active:
                proj.update()

        # Check collisions
        reward += self._check_collisions()

        # Check phase progression (all spawners destroyed)
        active_spawners = [s for s in self.spawners if s.active]
        if len(active_spawners) == 0 and len(self.spawners) > 0:
            reward += REWARDS['phase_progress']
            self.phase += 1
            if self.phase <= 5:
                self._spawn_phase_spawners()

        # Clean up inactive entities
        self.enemies = [e for e in self.enemies if e.active]
        self.projectiles = [p for p in self.projectiles if p.active]

        # Check end conditions
        if not self.player.active:
            reward += REWARDS['death']
            self.done = True
        elif self.steps >= MAX_STEPS_PER_EPISODE:
            self.done = True

        self.total_reward += reward
        return self._get_observation(), reward, self.done, self._get_info()

    def _check_collisions(self):
        """Check all collisions and return reward delta."""
        reward = 0

        # Player projectiles vs enemies
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

        # Player projectiles vs spawners
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

        # Enemies vs player (collision damage)
        if self.player.invulnerable <= 0:
            for enemy in self.enemies:
                if enemy.active and self.player.collides_with(enemy):
                    enemy.active = False  # Enemy dies on collision
                    self.player.take_damage(ENEMY['damage'])
                    self.player.invulnerable = PLAYER['invulnerability_frames']
                    reward += REWARDS['damage_taken']

        return reward

    def _get_observation(self):
        """Build fixed-size observation vector (22 features)."""
        obs = []

        # Player position (normalized to [0, 1])
        obs.append(self.player.x / WINDOW_WIDTH)
        obs.append(self.player.y / WINDOW_HEIGHT)

        # Player velocity (normalized to [-1, 1])
        obs.append(self.player.vx / PLAYER['max_velocity'])
        obs.append(self.player.vy / PLAYER['max_velocity'])

        # Player orientation (sin/cos for continuity)
        rad = math.radians(self.player.angle)
        obs.append(math.cos(rad))
        obs.append(math.sin(rad))

        # Nearest enemy: distance, direction, and speed
        nearest_enemy = self._find_nearest(self.enemies)
        if nearest_enemy:
            dist = self.player.distance_to(nearest_enemy) / DIAGONAL
            angle = self.player.angle_to(nearest_enemy)
            obs.append(dist)
            obs.append(math.cos(angle))
            obs.append(math.sin(angle))
            max_enemy_speed = ENEMY['speed'] * 1.5  # Max speed at phase 5
            obs.append(nearest_enemy.speed / max_enemy_speed)
        else:
            obs.extend([1.0, 0.0, 0.0, 0.0])  # No enemy

        # Nearest spawner: distance and direction
        active_spawners = [s for s in self.spawners if s.active]
        nearest_spawner = self._find_nearest(active_spawners)
        if nearest_spawner:
            dist = self.player.distance_to(nearest_spawner) / DIAGONAL
            angle = self.player.angle_to(nearest_spawner)
            obs.append(dist)
            obs.append(math.cos(angle))
            obs.append(math.sin(angle))
        else:
            obs.extend([1.0, 0.0, 0.0])  # No spawner

        # Player health (normalized to [0, 1])
        obs.append(self.player.health / PLAYER['max_health'])

        # Current phase (normalized to [0, 1])
        obs.append(self.phase / 5.0)

        # Enemy count (normalized, capped at 20)
        obs.append(min(len(self.enemies) / 20.0, 1.0))

        # Shoot ready (binary)
        obs.append(1.0 if self.player.can_shoot() else 0.0)

        # Wall distances (normalized to [0, 1])
        obs.append(self.player.x / WINDOW_WIDTH)  # Left
        obs.append((WINDOW_WIDTH - self.player.x) / WINDOW_WIDTH)  # Right
        obs.append(self.player.y / WINDOW_HEIGHT)  # Top
        obs.append((WINDOW_HEIGHT - self.player.y) / WINDOW_HEIGHT)  # Bottom

        # Active spawner count (normalized, max 5)
        obs.append(len(active_spawners) / 5.0)

        return obs

    def _find_nearest(self, entities):
        """Find nearest active entity from list."""
        nearest = None
        min_dist = float('inf')
        for e in entities:
            if e.active:
                dist = self.player.distance_to(e)
                if dist < min_dist:
                    min_dist = dist
                    nearest = e
        return nearest

    def _get_info(self):
        """Return additional info dict."""
        return {
            'phase': self.phase,
            'enemies_destroyed': self.enemies_destroyed,
            'spawners_destroyed': self.spawners_destroyed,
            'steps': self.steps,
            'player_health': self.player.health,
        }

    def render(self, info_text=""):
        """Render the current state using Pygame."""
        if not self.render_mode:
            return

        self.screen.fill(COLORS['background'])

        # Draw stars
        for x, y, brightness in self.stars:
            pygame.draw.circle(self.screen, (brightness, brightness, brightness), (x, y), 1)

        # Draw entities
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

        # Draw HUD
        self._draw_hud(info_text)

        pygame.display.flip()

    def _draw_hud(self, info_text):
        """Draw heads-up display."""
        # Health bar
        bar_x, bar_y = 10, 10
        bar_width, bar_height = 200, 20
        health_ratio = max(0, self.player.health / PLAYER['max_health'])

        pygame.draw.rect(self.screen, COLORS['health_bar_bg'],
                        (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, COLORS['health_bar'],
                        (bar_x, bar_y, bar_width * health_ratio, bar_height))
        pygame.draw.rect(self.screen, COLORS['text'],
                        (bar_x, bar_y, bar_width, bar_height), 2)

        health_text = self.font.render(f"HP: {max(0, self.player.health)}", True, COLORS['text'])
        self.screen.blit(health_text, (bar_x + 5, bar_y + 2))

        # Phase indicator
        phase_text = self.font_large.render(f"Phase {self.phase}", True, COLORS['phase_indicator'])
        self.screen.blit(phase_text, (WINDOW_WIDTH - 120, 10))

        # Score/stats
        stats_text = self.font.render(
            f"Enemies: {self.enemies_destroyed}  |  Spawners: {self.spawners_destroyed}  |  Score: {int(self.total_reward)}",
            True, COLORS['text']
        )
        self.screen.blit(stats_text, (10, 40))

        # Active spawners count
        active_spawners = sum(1 for s in self.spawners if s.active)
        spawner_text = self.font.render(
            f"Active Spawners: {active_spawners}  |  Enemies: {len(self.enemies)}",
            True, COLORS['text']
        )
        self.screen.blit(spawner_text, (10, 60))

        # Info text at bottom
        if info_text:
            info_surface = self.font.render(info_text, True, COLORS['text'])
            self.screen.blit(info_surface, (10, WINDOW_HEIGHT - 30))

    def close(self, quit_pygame=False):
        """Clean up Pygame resources."""
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
