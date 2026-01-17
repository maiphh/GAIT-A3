"""
Evaluation scripts for trained models.
Visual playback and performance metrics.
"""
import argparse
import os
import pygame
import numpy as np
from stable_baselines3 import PPO
from arena_gym import ArenaEnv


def evaluate_model(model_path, control_scheme, episodes=10, render=True, deterministic=True):
    """Evaluate a trained model and report metrics."""

    print(f"\n{'='*50}")
    print(f"Evaluating Model")
    print(f"{'='*50}")
    print(f"Model: {model_path}")
    print(f"Control scheme: {control_scheme}")
    print(f"Episodes: {episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*50}\n")

    # Load model
    model = PPO.load(model_path)

    # Create environment
    render_mode = 'human' if render else None
    env = ArenaEnv(control_scheme=control_scheme, render_mode=render_mode)

    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    phases_reached = []
    enemies_destroyed = []
    spawners_destroyed = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if render:
                env.arena.render(f"Evaluation Episode {ep+1}/{episodes} | ESC to skip")

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return None
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        done = True  # Skip to next episode

                pygame.time.Clock().tick(60)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        phases_reached.append(info.get('phase', 1))
        enemies_destroyed.append(info.get('enemies_destroyed', 0))
        spawners_destroyed.append(info.get('spawners_destroyed', 0))

        print(f"Episode {ep+1}: Reward={total_reward:.2f}, Steps={steps}, "
              f"Phase={info.get('phase', 1)}, Enemies={info.get('enemies_destroyed', 0)}, "
              f"Spawners={info.get('spawners_destroyed', 0)}")

    env.close()

    # Summary statistics
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Mean Reward:      {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Min/Max Reward:   {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    print(f"Mean Ep Length:   {np.mean(episode_lengths):.0f} +/- {np.std(episode_lengths):.0f}")
    print(f"Mean Phase:       {np.mean(phases_reached):.2f}")
    print(f"Max Phase:        {np.max(phases_reached)}")
    print(f"Mean Enemies:     {np.mean(enemies_destroyed):.1f}")
    print(f"Mean Spawners:    {np.mean(spawners_destroyed):.1f}")
    print(f"{'='*50}")

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'phases': phases_reached,
        'enemies': enemies_destroyed,
        'spawners': spawners_destroyed,
    }


def compare_models(rotation_path, directional_path, episodes=10):
    """Compare both control schemes side by side."""

    print(f"\n{'='*60}")
    print("COMPARING CONTROL SCHEMES")
    print(f"{'='*60}\n")

    print("Evaluating ROTATION control...")
    rot_results = evaluate_model(rotation_path, 'rotation', episodes, render=False)

    print("\nEvaluating DIRECTIONAL control...")
    dir_results = evaluate_model(directional_path, 'directional', episodes, render=False)

    if rot_results and dir_results:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Rotation':>15} {'Directional':>15}")
        print("-" * 50)
        print(f"{'Mean Reward':<20} {np.mean(rot_results['rewards']):>15.2f} {np.mean(dir_results['rewards']):>15.2f}")
        print(f"{'Mean Phase':<20} {np.mean(rot_results['phases']):>15.2f} {np.mean(dir_results['phases']):>15.2f}")
        print(f"{'Mean Enemies':<20} {np.mean(rot_results['enemies']):>15.1f} {np.mean(dir_results['enemies']):>15.1f}")
        print(f"{'Mean Spawners':<20} {np.mean(rot_results['spawners']):>15.1f} {np.mean(dir_results['spawners']):>15.1f}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained arena agent')
    parser.add_argument('--model', type=str, help='Path to model file (without .zip)')
    parser.add_argument('--scheme', type=str, choices=['rotation', 'directional'],
                       default='rotation', help='Control scheme')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions')
    parser.add_argument('--compare', action='store_true', help='Compare both control schemes')

    args = parser.parse_args()

    if args.compare:
        # Compare both schemes using default paths
        rot_path = 'models/ppo_rotation/ppo_rotation_final'
        dir_path = 'models/ppo_directional/ppo_directional_final'
        compare_models(rot_path, dir_path, args.episodes)
    else:
        # Evaluate single model
        if args.model is None:
            # Use default path based on scheme
            args.model = f'models/ppo_{args.scheme}/ppo_{args.scheme}_final'

        evaluate_model(
            model_path=args.model,
            control_scheme=args.scheme,
            episodes=args.episodes,
            render=not args.no_render,
            deterministic=not args.stochastic
        )


if __name__ == "__main__":
    main()
