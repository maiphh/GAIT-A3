"""
Training visualization utilities for Part 2 - Deep RL Arena.
Reads monitor.csv logs and displays training curves.
"""
import os
import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    """Visualizes training progress from monitor logs."""

    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir

    def load_monitor_data(self, scheme):
        """Load monitor.csv data for a control scheme."""
        path = os.path.join(self.log_dir, scheme, 'monitor.csv')
        if not os.path.exists(path):
            print(f"No log file found at {path}")
            return None

        rewards = []
        lengths = []
        times = []

        with open(path, 'r') as f:
            lines = f.readlines()[2:]  # Skip header

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 3:
                try:
                    rewards.append(float(parts[0]))
                    lengths.append(int(float(parts[1])))
                    times.append(float(parts[2]))
                except ValueError:
                    continue

        return {
            'rewards': np.array(rewards),
            'lengths': np.array(lengths),
            'times': np.array(times)
        }

    def moving_average(self, data, window=50):
        """Calculate moving average."""
        if len(data) < window:
            window = max(1, len(data))
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_scheme(self, scheme, save_path=None):
        """Plot training curves for a single control scheme."""
        data = self.load_monitor_data(scheme)
        if data is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Training Progress - {scheme.title()} Control', fontsize=14)

        rewards = data['rewards']
        lengths = data['lengths']

        # Reward curve
        ax1 = axes[0, 0]
        ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) > 10:
            ma = self.moving_average(rewards)
            ax1.plot(range(len(ma)), ma, color='blue', linewidth=2, label='Moving Avg (50)')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Episode length curve
        ax2 = axes[0, 1]
        ax2.plot(lengths, alpha=0.3, color='green', label='Raw')
        if len(lengths) > 10:
            ma = self.moving_average(lengths)
            ax2.plot(range(len(ma)), ma, color='green', linewidth=2, label='Moving Avg (50)')
        ax2.axhline(y=3000, color='red', linestyle='--', alpha=0.5, label='Max Steps')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Survival rate (episodes reaching 3000 steps)
        ax3 = axes[1, 0]
        survival = (lengths == 3000).astype(float)
        if len(survival) >= 50:
            survival_rate = self.moving_average(survival, 50)
            ax3.plot(range(len(survival_rate)), survival_rate * 100, color='purple', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Survival Rate (%)')
        ax3.set_title('Survival Rate (50-episode window)')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        # Positive reward rate
        ax4 = axes[1, 1]
        positive = (rewards > 0).astype(float)
        if len(positive) >= 50:
            positive_rate = self.moving_average(positive, 50)
            ax4.plot(range(len(positive_rate)), positive_rate * 100, color='orange', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Positive Reward Rate (%)')
        ax4.set_title('Episodes with Positive Reward (50-episode window)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")

        return fig

    def plot_comparison(self, save_path=None):
        """Compare both control schemes on the same plot."""
        rot_data = self.load_monitor_data('rotation')
        dir_data = self.load_monitor_data('directional')

        if rot_data is None and dir_data is None:
            print("No training data found for either scheme.")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Control Scheme Comparison', fontsize=14)

        # Reward comparison
        ax1 = axes[0]
        if rot_data is not None and len(rot_data['rewards']) > 10:
            ma = self.moving_average(rot_data['rewards'])
            ax1.plot(range(len(ma)), ma, color='blue', linewidth=2, label='Rotation')
        if dir_data is not None and len(dir_data['rewards']) > 10:
            ma = self.moving_average(dir_data['rewards'])
            ax1.plot(range(len(ma)), ma, color='red', linewidth=2, label='Directional')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward (Moving Avg)')
        ax1.set_title('Reward Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Survival rate comparison
        ax2 = axes[1]
        if rot_data is not None:
            survival = (rot_data['lengths'] == 3000).astype(float)
            if len(survival) >= 50:
                sr = self.moving_average(survival, 50)
                ax2.plot(range(len(sr)), sr * 100, color='blue', linewidth=2, label='Rotation')
        if dir_data is not None:
            survival = (dir_data['lengths'] == 3000).astype(float)
            if len(survival) >= 50:
                sr = self.moving_average(survival, 50)
                ax2.plot(range(len(sr)), sr * 100, color='red', linewidth=2, label='Directional')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Survival Rate (%)')
        ax2.set_title('Survival Rate Comparison')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")

        return fig

    def print_summary(self, scheme):
        """Print training summary statistics."""
        data = self.load_monitor_data(scheme)
        if data is None:
            return

        rewards = data['rewards']
        lengths = data['lengths']

        print(f"\n{'='*50}")
        print(f"TRAINING SUMMARY - {scheme.upper()}")
        print(f"{'='*50}")
        print(f"Total Episodes: {len(rewards)}")
        print(f"Mean Reward: {np.mean(rewards):.2f}")
        print(f"Max Reward: {np.max(rewards):.2f}")
        print(f"Min Reward: {np.min(rewards):.2f}")
        print(f"Std Reward: {np.std(rewards):.2f}")
        print(f"Mean Episode Length: {np.mean(lengths):.0f}")

        survive_count = np.sum(lengths == 3000)
        positive_count = np.sum(rewards > 0)
        print(f"Episodes Surviving (3000 steps): {survive_count} ({survive_count/len(rewards)*100:.1f}%)")
        print(f"Episodes with Positive Reward: {positive_count} ({positive_count/len(rewards)*100:.1f}%)")

        if len(rewards) >= 50:
            print(f"\nLast 50 Episodes:")
            print(f"  Mean Reward: {np.mean(rewards[-50:]):.2f}")
            print(f"  Max Reward: {np.max(rewards[-50:]):.2f}")
        print(f"{'='*50}")


def show_training_graphs(log_dir='logs'):
    """Display training graphs with interactive menu."""
    viz = TrainingVisualizer(log_dir)

    print("\nTraining Visualization")
    print("=" * 30)
    print("1. Rotation scheme graphs")
    print("2. Directional scheme graphs")
    print("3. Compare both schemes")
    print("4. Print summaries")
    print("5. Cancel")

    choice = input("\nSelect option (1-5): ").strip()

    if choice == '1':
        viz.print_summary('rotation')
        fig = viz.plot_scheme('rotation')
        if fig:
            plt.show()
    elif choice == '2':
        viz.print_summary('directional')
        fig = viz.plot_scheme('directional')
        if fig:
            plt.show()
    elif choice == '3':
        fig = viz.plot_comparison()
        if fig:
            plt.show()
    elif choice == '4':
        viz.print_summary('rotation')
        viz.print_summary('directional')
    elif choice == '5':
        print("Cancelled.")
    else:
        print("Invalid option.")


if __name__ == "__main__":
    show_training_graphs()
