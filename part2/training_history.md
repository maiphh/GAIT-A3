# Training History - Deep RL Arena

## Run 2 - 2026-01-07 (Planned)

### Changes from Run 1
- Expanded observation space (16 → 22 features)
- Increased training timesteps (500k → 1.5M)
- Adjusted hyperparameters for better exploration and stability
- Increased survival reward to encourage staying alive

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| total_timesteps | 1,500,000 |
| learning_rate | 1e-4 |
| n_steps | 4096 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.02 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| policy_network | [256, 256] |

### Reward Structure
| Event | Reward |
|-------|--------|
| destroy_enemy | +15.0 |
| destroy_spawner | +50.0 |
| phase_progress | +100.0 |
| damage_taken | -5.0 |
| death | -200.0 |
| survival_tick | +0.1 |

### Observation Space (22 features)
1. Player x position (normalized)
2. Player y position (normalized)
3. Player velocity x (normalized)
4. Player velocity y (normalized)
5. Player angle cos
6. Player angle sin
7. Nearest enemy distance (normalized)
8. Nearest enemy direction cos
9. Nearest enemy direction sin
10. Nearest enemy speed (normalized)
11. Nearest spawner distance (normalized)
12. Nearest spawner direction cos
13. Nearest spawner direction sin
14. Player health (normalized)
15. Current phase (normalized)
16. Enemy count (normalized, capped at 20)
17. Shoot ready (binary)
18. Distance to left wall (normalized)
19. Distance to right wall (normalized)
20. Distance to top wall (normalized)
21. Distance to bottom wall (normalized)
22. Active spawner count (normalized)

### Results
- Status: Pending
- Rotation: TBD
- Directional: TBD

---

## Run 1 - 2026-01-07 (Kaggle)

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| total_timesteps | 500,000 |
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| policy_network | [128, 128] |

### Reward Structure
| Event | Reward |
|-------|--------|
| destroy_enemy | +10.0 |
| destroy_spawner | +50.0 |
| phase_progress | +100.0 |
| damage_taken | -5.0 |
| death | -200.0 |
| survival_tick | +0.01 |

### Observation Space (16 features)
1. Player x position (normalized)
2. Player y position (normalized)
3. Player velocity x (normalized)
4. Player velocity y (normalized)
5. Player angle cos
6. Player angle sin
7. Nearest enemy distance (normalized)
8. Nearest enemy direction cos
9. Nearest enemy direction sin
10. Nearest spawner distance (normalized)
11. Nearest spawner direction cos
12. Nearest spawner direction sin
13. Player health (normalized)
14. Current phase (normalized)
15. Enemy count (normalized, capped at 20)
16. Shoot ready (binary)

### Results
| Metric | Rotation | Directional |
|--------|----------|-------------|
| Episodes | 322 | 286 |
| Mean Reward | -212.70 | -168.82 |
| Max Reward | 25.00 | 205.00 |
| Survival Rate | 8.7% | 14.3% |
| Positive Reward Rate | 2.5% | 9.1% |
| Death Rate | 89.8% | 71.0% |
| Good Runs (r >= 50) | 0.0% | 2.1% |

### Analysis
- Both schemes still in early learning phase
- Directional significantly outperforms Rotation
- High death rate indicates agents struggling to survive
- Positive improvement trend observed (first 100 vs last 100 episodes)
- Need more training steps for convergence
