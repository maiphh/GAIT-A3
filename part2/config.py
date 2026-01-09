"""
Configuration file for the Deep RL Arena.
All configurable parameters are centralized here.
"""
import math

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 680
FPS = 60
RENDER_DURING_TRAINING = False

# =============================================================================
# COLORS (RGB) - Dark space theme
# =============================================================================
COLORS = {
    'background': (15, 15, 25),
    'text': (255, 255, 255),
    'button': (70, 130, 180),
    'button_hover': (100, 160, 210),
    'player': (100, 180, 255),
    'enemy': (255, 80, 80),
    'spawner': (148, 0, 211),
    'projectile_player': (255, 255, 100),
    'projectile_enemy': (255, 100, 100),
    'health_bar': (100, 255, 100),
    'health_bar_bg': (60, 60, 60),
    'phase_indicator': (255, 215, 0),
}

# =============================================================================
# PLAYER SETTINGS
# =============================================================================
PLAYER = {
    'start_x': WINDOW_WIDTH // 2,
    'start_y': WINDOW_HEIGHT // 2,
    'max_health': 100,
    'speed': 5.0,
    'thrust_power': 0.3,
    'rotation_speed': 5.0,
    'max_velocity': 8.0,
    'friction': 0.98,
    'shoot_cooldown': 10,
    'size': 20,
    'invulnerability_frames': 30,
}

# =============================================================================
# ENEMY SETTINGS
# =============================================================================
ENEMY = {
    'health': 10,
    'speed': 2.5,
    'size': 15,
    'damage': 10,
}

# =============================================================================
# SPAWNER SETTINGS
# =============================================================================
SPAWNER = {
    'health': 40,
    'size': 30,
    'spawn_interval': 120,
    'max_enemies_per_spawner': 10,
}

# =============================================================================
# PROJECTILE SETTINGS
# =============================================================================
PROJECTILE = {
    'speed': 12.0,
    'size': 5,
    'damage': 10,
    'lifetime': 60,
}

# =============================================================================
# PHASE SYSTEM
# =============================================================================
PHASES = {
    1: {'spawners': 1, 'spawn_rate': 1.0, 'enemy_speed': 1.0},
    2: {'spawners': 2, 'spawn_rate': 1.2, 'enemy_speed': 1.1},
    3: {'spawners': 3, 'spawn_rate': 1.4, 'enemy_speed': 1.2},
    4: {'spawners': 4, 'spawn_rate': 1.6, 'enemy_speed': 1.3},
    5: {'spawners': 5, 'spawn_rate': 2.0, 'enemy_speed': 1.5},
}

# =============================================================================
# REWARD STRUCTURE
# =============================================================================
REWARDS = {
    'destroy_enemy': 10.0,
    'destroy_spawner': 80.0,
    'phase_progress': 10000.0,
    'damage_taken': -5.0,
    'death': -200.0,
    'survival_tick': 0.01,
    'hit_enemy': 0.0,      # 0 because enemies die in 1 hit (10 HP, 10 damage)
    'hit_spawner': 5.0,
}

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
TRAINING = {
    'total_timesteps': 1500000,
    'learning_rate': 1e-4,
    'n_steps': 4096,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.02,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'policy_network': [256, 256],
}

MAX_STEPS_PER_EPISODE = 3000

# =============================================================================
# ACTION SPACES
# =============================================================================
# Control Scheme 1: Rotation-based
ACTIONS_ROTATION = {
    0: 'no_action',
    1: 'thrust',
    2: 'rotate_left',
    3: 'rotate_right',
    4: 'shoot',
}
NUM_ACTIONS_ROTATION = 5

# Control Scheme 2: Directional
ACTIONS_DIRECTIONAL = {
    0: 'no_action',
    1: 'move_up',
    2: 'move_down',
    3: 'move_left',
    4: 'move_right',
    5: 'shoot',
}
NUM_ACTIONS_DIRECTIONAL = 6

# =============================================================================
# OBSERVATION SPACE
# =============================================================================
OBSERVATION_SIZE = 23
DIAGONAL = math.sqrt(WINDOW_WIDTH**2 + WINDOW_HEIGHT**2)
