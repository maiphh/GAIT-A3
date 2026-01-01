"""
Training and visualization utilities.
"""
import matplotlib.pyplot as plt
import numpy as np
import os


class TrainingStats:
    """Tracks and visualizes training progress."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_success = []
    
    def record_episode(self, reward, steps, success):
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.episode_success.append(1 if success else 0)
    
    def get_moving_average(self, data, window=50):
        if len(data) < window:
            window = max(1, len(data))
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def plot_training_curves(self, title="Training Progress", save_path=None):
        """Generate training curve plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=14)
        
        # Reward curve
        ax1 = axes[0, 0]
        ax1.plot(self.episode_rewards, alpha=0.3, color='blue', label='Raw')
        if len(self.episode_rewards) > 10:
            ma = self.get_moving_average(self.episode_rewards)
            ax1.plot(range(len(ma)), ma, color='blue', linewidth=2, label='Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Steps curve
        ax2 = axes[0, 1]
        ax2.plot(self.episode_steps, alpha=0.3, color='green', label='Raw')
        if len(self.episode_steps) > 10:
            ma = self.get_moving_average(self.episode_steps)
            ax2.plot(range(len(ma)), ma, color='green', linewidth=2, label='Moving Avg')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Success rate
        ax3 = axes[1, 0]
        if len(self.episode_success) >= 50:
            success_rate = self.get_moving_average(self.episode_success, 50)
            ax3.plot(range(len(success_rate)), success_rate, color='purple', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate (50-episode window)')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # Cumulative reward
        ax4 = axes[1, 1]
        cumulative = np.cumsum(self.episode_rewards)
        ax4.plot(cumulative, color='orange', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Cumulative Reward')
        ax4.set_title('Cumulative Reward')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved training curves to {save_path}")
        
        plt.show()
        return fig
    
    def print_summary(self):
        """Print training summary statistics."""
        if not self.episode_rewards:
            print("No training data available.")
            return
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Average Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Max Reward: {max(self.episode_rewards):.2f}")
        print(f"Average Steps: {np.mean(self.episode_steps):.1f}")
        print(f"Success Rate: {np.mean(self.episode_success)*100:.1f}%")
        
        # Last 50 episodes
        if len(self.episode_rewards) >= 50:
            print("\nLast 50 Episodes:")
            print(f"  Average Reward: {np.mean(self.episode_rewards[-50:]):.2f}")
            print(f"  Success Rate: {np.mean(self.episode_success[-50:])*100:.1f}%")
        print("="*50)


def compare_training_curves(stats_list, labels, title="Algorithm Comparison", save_path=None):
    """Compare multiple training runs on the same plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    ax1 = axes[0]
    for i, (stats, label) in enumerate(zip(stats_list, labels)):
        color = colors[i % len(colors)]
        if len(stats.episode_rewards) > 10:
            ma = stats.get_moving_average(stats.episode_rewards)
            ax1.plot(range(len(ma)), ma, color=color, linewidth=2, label=label)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Reward Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for i, (stats, label) in enumerate(zip(stats_list, labels)):
        color = colors[i % len(colors)]
        if len(stats.episode_success) >= 50:
            success_rate = stats.get_moving_average(stats.episode_success, 50)
            ax2.plot(range(len(success_rate)), success_rate, color=color, linewidth=2, label=label)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate Comparison')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison to {save_path}")
    
    plt.show()
    return fig
