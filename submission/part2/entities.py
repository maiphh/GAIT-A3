"""
Game entities for the Arena environment.
Player, Enemy, Spawner, and Projectile classes.
"""
import pygame
import math
import random
from config import (
    PLAYER, ENEMY, SPAWNER, PROJECTILE,
    WINDOW_WIDTH, WINDOW_HEIGHT, COLORS
)


class Entity:
    """Base class for all arena entities."""

    def __init__(self, x, y, size, health=1):
        self.x = x
        self.y = y
        self.size = size
        self.health = health
        self.max_health = health
        self.active = True

    def get_position(self):
        return (self.x, self.y)

    def distance_to(self, other):
        """Calculate Euclidean distance to another entity."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def angle_to(self, other):
        """Calculate angle to another entity (radians)."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.atan2(dy, dx)

    def collides_with(self, other):
        """Check collision with another entity."""
        return self.distance_to(other) < (self.size + other.size)

    def take_damage(self, amount):
        """Apply damage and return True if destroyed."""
        self.health -= amount
        if self.health <= 0:
            self.active = False
            return True
        return False


class Player(Entity):
    """Player-controlled ship entity."""

    def __init__(self, x=None, y=None):
        x = x if x is not None else PLAYER['start_x']
        y = y if y is not None else PLAYER['start_y']
        super().__init__(x, y, PLAYER['size'], PLAYER['max_health'])

        # Movement state
        self.vx = 0.0
        self.vy = 0.0
        self.angle = -90  # Facing up initially (degrees)

        # Combat state
        self.shoot_cooldown = 0
        self.invulnerable = 0

        # Control scheme
        self.control_scheme = 'rotation'

    def reset(self):
        """Reset player to initial state."""
        self.x = PLAYER['start_x']
        self.y = PLAYER['start_y']
        self.vx = 0.0
        self.vy = 0.0
        self.angle = -90
        self.health = PLAYER['max_health']
        self.max_health = PLAYER['max_health']
        self.shoot_cooldown = 0
        self.invulnerable = 0
        self.active = True

    def apply_action_rotation(self, action):
        """Apply rotation-based control action. Returns True if shoot requested."""
        shoot = False

        if action == 1:  # Thrust
            rad = math.radians(self.angle)
            self.vx += math.cos(rad) * PLAYER['thrust_power']
            self.vy += math.sin(rad) * PLAYER['thrust_power']
        elif action == 2:  # Rotate left
            self.angle -= PLAYER['rotation_speed']
        elif action == 3:  # Rotate right
            self.angle += PLAYER['rotation_speed']
        elif action == 4:  # Shoot
            shoot = True

        return shoot

    def apply_action_directional(self, action):
        """Apply directional control action. Returns True if shoot requested."""
        shoot = False

        if action == 1:  # Move up
            self.vy = -PLAYER['speed']
        elif action == 2:  # Move down
            self.vy = PLAYER['speed']
        elif action == 3:  # Move left
            self.vx = -PLAYER['speed']
        elif action == 4:  # Move right
            self.vx = PLAYER['speed']
        elif action == 5:  # Shoot
            shoot = True

        return shoot

    def update(self):
        """Update player physics."""
        # Apply friction
        if self.control_scheme == 'rotation':
            self.vx *= PLAYER['friction']
            self.vy *= PLAYER['friction']
        else:
            # Directional mode: faster deceleration
            self.vx *= 0.8
            self.vy *= 0.8

        # Clamp velocity
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > PLAYER['max_velocity']:
            scale = PLAYER['max_velocity'] / speed
            self.vx *= scale
            self.vy *= scale

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Keep in bounds
        self.x = max(self.size, min(WINDOW_WIDTH - self.size, self.x))
        self.y = max(self.size, min(WINDOW_HEIGHT - self.size, self.y))

        # Update cooldowns
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.invulnerable > 0:
            self.invulnerable -= 1

    def can_shoot(self):
        return self.shoot_cooldown <= 0

    def shoot(self):
        """Fire a projectile, return Projectile or None."""
        if not self.can_shoot():
            return None

        self.shoot_cooldown = PLAYER['shoot_cooldown']

        # For directional mode, always shoot upward
        if self.control_scheme == 'directional':
            rad = math.radians(-90)  # Up
        else:
            rad = math.radians(self.angle)

        # Spawn projectile at ship front
        px = self.x + math.cos(rad) * (self.size + 5)
        py = self.y + math.sin(rad) * (self.size + 5)

        return Projectile(px, py, rad, owner='player')

    def draw(self, screen):
        """Draw the player ship."""
        # Draw ship as triangle pointing in angle direction
        if self.control_scheme == 'directional':
            draw_angle = -90  # Always point up
        else:
            draw_angle = self.angle

        rad = math.radians(draw_angle)

        # Triangle points
        front = (self.x + math.cos(rad) * self.size,
                 self.y + math.sin(rad) * self.size)
        left = (self.x + math.cos(rad + 2.5) * self.size * 0.7,
                self.y + math.sin(rad + 2.5) * self.size * 0.7)
        right = (self.x + math.cos(rad - 2.5) * self.size * 0.7,
                 self.y + math.sin(rad - 2.5) * self.size * 0.7)

        color = COLORS['player']
        if self.invulnerable > 0 and (self.invulnerable // 3) % 2:
            color = (200, 200, 255)  # Flash when invulnerable

        pygame.draw.polygon(screen, color, [front, left, right])


class Enemy(Entity):
    """Enemy that navigates toward the player."""

    def __init__(self, x, y, spawner_id=None):
        super().__init__(x, y, ENEMY['size'], ENEMY['health'])
        self.spawner_id = spawner_id
        self.speed = ENEMY['speed']

    def update(self, player):
        """Move toward player."""
        if not self.active or not player.active:
            return

        # Calculate direction to player
        angle = self.angle_to(player)

        # Move toward player
        self.x += math.cos(angle) * self.speed
        self.y += math.sin(angle) * self.speed

        # Keep in bounds
        self.x = max(self.size, min(WINDOW_WIDTH - self.size, self.x))
        self.y = max(self.size, min(WINDOW_HEIGHT - self.size, self.y))

    def draw(self, screen):
        """Draw the enemy as a red circle with eyes."""
        pygame.draw.circle(screen, COLORS['enemy'],
                          (int(self.x), int(self.y)), self.size)
        # Simple alien eyes
        pygame.draw.circle(screen, (255, 255, 255),
                          (int(self.x - 5), int(self.y - 3)), 4)
        pygame.draw.circle(screen, (255, 255, 255),
                          (int(self.x + 5), int(self.y - 3)), 4)
        # Pupils
        pygame.draw.circle(screen, (0, 0, 0),
                          (int(self.x - 5), int(self.y - 3)), 2)
        pygame.draw.circle(screen, (0, 0, 0),
                          (int(self.x + 5), int(self.y - 3)), 2)


class Spawner(Entity):
    """Enemy spawner that creates enemies periodically."""

    def __init__(self, x, y, spawner_id):
        super().__init__(x, y, SPAWNER['size'], SPAWNER['health'])
        self.spawner_id = spawner_id
        self.spawn_timer = 0
        self.spawn_interval = SPAWNER['spawn_interval']
        self.enemies_spawned = 0
        self.max_enemies = SPAWNER['max_enemies_per_spawner']

    def update(self, current_enemy_count=0):
        """Update spawn timer. Returns new Enemy or None."""
        if not self.active:
            return None

        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            # Limit total enemies from this spawner
            if self.enemies_spawned < self.max_enemies:
                return self.spawn_enemy()
        return None

    def spawn_enemy(self):
        """Create a new enemy at spawner location."""
        # Spawn at random offset from spawner
        offset_angle = random.random() * 2 * math.pi
        offset_dist = self.size + ENEMY['size'] + 10

        ex = self.x + math.cos(offset_angle) * offset_dist
        ey = self.y + math.sin(offset_angle) * offset_dist

        # Keep spawn position in bounds
        ex = max(ENEMY['size'], min(WINDOW_WIDTH - ENEMY['size'], ex))
        ey = max(ENEMY['size'], min(WINDOW_HEIGHT - ENEMY['size'], ey))

        self.enemies_spawned += 1
        return Enemy(ex, ey, self.spawner_id)

    def draw(self, screen):
        """Draw the spawner as a purple hexagon."""
        points = []
        for i in range(6):
            angle = i * math.pi / 3 - math.pi / 6
            px = self.x + math.cos(angle) * self.size
            py = self.y + math.sin(angle) * self.size
            points.append((px, py))

        pygame.draw.polygon(screen, COLORS['spawner'], points)
        pygame.draw.polygon(screen, (200, 100, 255), points, 2)  # Border

        # Health bar
        bar_width = self.size * 2
        bar_height = 6
        health_ratio = self.health / self.max_health

        bar_x = self.x - bar_width / 2
        bar_y = self.y - self.size - 12

        pygame.draw.rect(screen, COLORS['health_bar_bg'],
                        (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(screen, COLORS['health_bar'],
                        (bar_x, bar_y, bar_width * health_ratio, bar_height))


class Projectile(Entity):
    """Projectile fired by player or enemy."""

    def __init__(self, x, y, angle, owner='player'):
        super().__init__(x, y, PROJECTILE['size'])
        self.angle = angle  # In radians
        self.owner = owner
        self.speed = PROJECTILE['speed']
        self.damage = PROJECTILE['damage']
        self.lifetime = PROJECTILE['lifetime']

    def update(self):
        """Move projectile forward."""
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.lifetime -= 1

        # Deactivate if out of bounds or expired
        if (self.x < -self.size or self.x > WINDOW_WIDTH + self.size or
            self.y < -self.size or self.y > WINDOW_HEIGHT + self.size or
            self.lifetime <= 0):
            self.active = False

    def draw(self, screen):
        """Draw the projectile."""
        color = COLORS['projectile_player'] if self.owner == 'player' else COLORS['projectile_enemy']
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)
        # Add glow effect
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.size - 2)
