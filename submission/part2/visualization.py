"""
Training visualization for Deep RL Arena.
Reads monitor.csv and shows training curves with phase analysis.
"""
import os
import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    """Visualizes training progress from monitor logs."""

    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir

    def load_monitor_data(self, scheme):
        """Load monitor.csv data."""
        path = os.path.join(self.log_dir, scheme, 'monitor.csv')
        if not os.path.exists(path):
            print(f"No log file at {path}")
            return None

        rewards, lengths, times = [], [], []

        with open(path, 'r') as f:
            lines = f.readlines()[2:]  # skip header

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
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

    def estimate_phase(self, reward):
        """Estimate max phase from reward. 10k per phase, 100k victory."""
        if reward < 0:
            return 0
        if reward >= 150000:
            return 5
        return min(5, int(reward / 10000) + 1)

    def moving_avg(self, data, window=50):
        """Simple moving average."""
        if len(data) < window:
            window = max(1, len(data))
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_scheme(self, scheme, save_path=None):
        """Plot training curves for one scheme."""
        data = self.load_monitor_data(scheme)
        if data is None:
            return None

        rewards = data['rewards']
        lengths = data['lengths']
        n = len(rewards)

        if n == 0:
            print(f"No episodes for {scheme}")
            return None

        phases = np.array([self.estimate_phase(r) for r in rewards])

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Training - {scheme.title()} ({n} episodes)', fontsize=14)

        # --- reward curve ---
        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.3, color='blue')
        if n > 10:
            ma = self.moving_avg(rewards)
            ax.plot(range(len(ma)), ma, color='blue', linewidth=2, label='MA50')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'Rewards (mean: {np.mean(rewards):.0f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- episode length ---
        ax = axes[0, 1]
        ax.plot(lengths, alpha=0.3, color='green')
        if n > 10:
            ma = self.moving_avg(lengths)
            ax.plot(range(len(ma)), ma, color='green', linewidth=2)
        ax.axhline(y=3000, color='red', linestyle='--', alpha=0.5, label='Max')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title(f'Episode Length (mean: {np.mean(lengths):.0f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- phase reached ---
        ax = axes[0, 2]
        ax.plot(phases, alpha=0.3, color='purple')
        if n > 10:
            ma = self.moving_avg(phases)
            ax.plot(range(len(ma)), ma, color='purple', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Max Phase')
        ax.set_title(f'Phase Reached (mean: {np.mean(phases):.1f})')
        ax.set_ylim(0, 5.5)
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.3)

        # --- phase distribution ---
        ax = axes[1, 0]
        counts = [np.sum(phases == i) for i in range(6)]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'blue']
        bars = ax.bar(range(6), counts, color=colors)
        ax.set_xlabel('Max Phase')
        ax.set_ylabel('Episodes')
        ax.set_title('Phase Distribution')
        ax.set_xticks(range(6))
        ax.set_xticklabels(['Death', 'P1', 'P2', 'P3', 'P4', 'P5'])
        for i, (bar, cnt) in enumerate(zip(bars, counts)):
            if cnt > 0:
                pct = cnt / n * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + n*0.01,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        # --- phase rate over time ---
        ax = axes[1, 1]
        window = min(100, n // 4) if n > 20 else n
        if window > 0:
            p2 = self.moving_avg((phases >= 2).astype(float), window) * 100
            p3 = self.moving_avg((phases >= 3).astype(float), window) * 100
            p4 = self.moving_avg((phases >= 4).astype(float), window) * 100

            x = range(len(p2))
            ax.plot(x, p2, label='Phase 2+', color='yellow', linewidth=2)
            ax.plot(x, p3, label='Phase 3+', color='green', linewidth=2)
            ax.plot(x, p4, label='Phase 4+', color='blue', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Rate (%)')
            ax.set_title(f'Phase Rate ({window}-ep window)')
            ax.set_ylim(0, 100)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # --- death rate ---
        ax = axes[1, 2]
        deaths = (lengths < 3000).astype(float)
        if n > 10:
            dr = self.moving_avg(deaths, min(50, n)) * 100
            ax.plot(range(len(dr)), dr, color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Death Rate (%)')
        total_deaths = np.sum(deaths)
        ax.set_title(f'Deaths ({total_deaths:.0f}/{n} = {total_deaths/n*100:.1f}%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")

        return fig

    def plot_comparison(self, save_path=None):
        """Compare both schemes."""
        rot = self.load_monitor_data('rotation')
        dir = self.load_monitor_data('directional')

        if rot is None and dir is None:
            print("No data found.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Control Scheme Comparison', fontsize=14)

        # --- reward ---
        ax = axes[0, 0]
        if rot is not None and len(rot['rewards']) > 10:
            ma = self.moving_avg(rot['rewards'])
            ax.plot(range(len(ma)), ma, color='blue', linewidth=2, label='Rotation')
        if dir is not None and len(dir['rewards']) > 10:
            ma = self.moving_avg(dir['rewards'])
            ax.plot(range(len(ma)), ma, color='red', linewidth=2, label='Directional')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (MA50)')
        ax.set_title('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- phase ---
        ax = axes[0, 1]
        if rot is not None and len(rot['rewards']) > 10:
            phases = np.array([self.estimate_phase(r) for r in rot['rewards']])
            ma = self.moving_avg(phases)
            ax.plot(range(len(ma)), ma, color='blue', linewidth=2, label='Rotation')
        if dir is not None and len(dir['rewards']) > 10:
            phases = np.array([self.estimate_phase(r) for r in dir['rewards']])
            ma = self.moving_avg(phases)
            ax.plot(range(len(ma)), ma, color='red', linewidth=2, label='Directional')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Max Phase (MA50)')
        ax.set_title('Phase Progression')
        ax.set_ylim(0, 5.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- phase 4+ rate ---
        ax = axes[1, 0]
        window = 100
        if rot is not None and len(rot['rewards']) > window:
            phases = np.array([self.estimate_phase(r) for r in rot['rewards']])
            p4 = self.moving_avg((phases >= 4).astype(float), window) * 100
            ax.plot(range(len(p4)), p4, color='blue', linewidth=2, label='Rotation')
        if dir is not None and len(dir['rewards']) > window:
            phases = np.array([self.estimate_phase(r) for r in dir['rewards']])
            p4 = self.moving_avg((phases >= 4).astype(float), window) * 100
            ax.plot(range(len(p4)), p4, color='red', linewidth=2, label='Directional')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Phase 4+ Rate (%)')
        ax.set_title('Phase 4+ Rate (100-ep)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- death rate ---
        ax = axes[1, 1]
        window = 50
        if rot is not None and len(rot['lengths']) > window:
            deaths = (rot['lengths'] < 3000).astype(float)
            dr = self.moving_avg(deaths, window) * 100
            ax.plot(range(len(dr)), dr, color='blue', linewidth=2, label='Rotation')
        if dir is not None and len(dir['lengths']) > window:
            deaths = (dir['lengths'] < 3000).astype(float)
            dr = self.moving_avg(deaths, window) * 100
            ax.plot(range(len(dr)), dr, color='red', linewidth=2, label='Directional')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Death Rate (%)')
        ax.set_title('Death Rate (50-ep)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")

        return fig

    def print_summary(self, scheme):
        """Print training summary."""
        data = self.load_monitor_data(scheme)
        if data is None:
            return

        rewards = data['rewards']
        lengths = data['lengths']
        n = len(rewards)

        if n == 0:
            print(f"No episodes for {scheme}")
            return

        phases = np.array([self.estimate_phase(r) for r in rewards])

        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY - {scheme.upper()}")
        print(f"{'='*60}")
        print(f"Total Episodes: {n}")
        print(f"\n--- Rewards ---")
        print(f"  Mean: {np.mean(rewards):.0f}")
        print(f"  Std:  {np.std(rewards):.0f}")
        print(f"  Max:  {np.max(rewards):.0f}")
        print(f"  Min:  {np.min(rewards):.0f}")

        print(f"\n--- Episode Length ---")
        print(f"  Mean: {np.mean(lengths):.0f}")
        print(f"  Max:  {np.max(lengths):.0f}")

        print(f"\n--- Phase Distribution ---")
        for p in range(6):
            cnt = np.sum(phases == p)
            pct = cnt / n * 100
            label = 'Death' if p == 0 else f'Phase {p}'
            print(f"  {label}: {cnt} ({pct:.1f}%)")

        print(f"\n--- Phase Rates ---")
        for p in [2, 3, 4, 5]:
            cnt = np.sum(phases >= p)
            pct = cnt / n * 100
            print(f"  Phase {p}+: {cnt} ({pct:.1f}%)")

        deaths = np.sum(lengths < 3000)
        print(f"\n--- Survival ---")
        print(f"  Deaths: {deaths} ({deaths/n*100:.1f}%)")
        print(f"  Survived: {n - deaths} ({(n-deaths)/n*100:.1f}%)")

        # last 100
        if n >= 100:
            last = 100
            print(f"\n--- Last {last} Episodes ---")
            print(f"  Mean Reward: {np.mean(rewards[-last:]):.0f}")
            print(f"  Mean Phase: {np.mean(phases[-last:]):.2f}")
            for p in [2, 3, 4]:
                cnt = np.sum(phases[-last:] >= p)
                print(f"  Phase {p}+: {cnt}%")

        print(f"{'='*60}")


def show_training_graphs(log_dir='logs'):
    """Interactive menu for viewing graphs."""
    viz = TrainingVisualizer(log_dir)

    print("\n" + "="*40)
    print("Training Visualization")
    print("="*40)
    print("1. Rotation graphs")
    print("2. Directional graphs")
    print("3. Compare both")
    print("4. Print summaries")
    print("5. Exit")

    choice = input("\nSelect (1-5): ").strip()

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
        viz.print_summary('rotation')
        viz.print_summary('directional')
        fig = viz.plot_comparison()
        if fig:
            plt.show()
    elif choice == '4':
        viz.print_summary('rotation')
        viz.print_summary('directional')
    else:
        print("Exited.")


if __name__ == "__main__":
    show_training_graphs()
