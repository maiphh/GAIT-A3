"""
Base Agent class for temporal difference learning algorithms.
Shared functionality between Q-Learning and SARSA.
"""
import random
import math
from collections import defaultdict
from config import TRAINING, NUM_ACTIONS, INTRINSIC_SCALE


class BaseAgent:
    """Base class for RL agents with epsilon-greedy exploration."""
    
    def __init__(self, alpha=TRAINING['alpha'], gamma=TRAINING['gamma'], epsilon_start=TRAINING['epsilon_start'], epsilon_end=TRAINING['epsilon_end'], episodes=TRAINING['episodes'], use_intrinsic=False, scale=INTRINSIC_SCALE):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.total_episodes = episodes
        
        self.q_table = defaultdict(lambda: [0.0] * NUM_ACTIONS)
        self.current_episode = 0

        self.use_intrinsic = use_intrinsic
        self.scale = scale
        self.visit_counts = defaultdict(int)
    
    @property
    def epsilon(self):
        """Linear decay from epsilon_start to epsilon_end."""
        if self.total_episodes <= 1:
            return self.epsilon_end
        progress = self.current_episode / (self.total_episodes - 1)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
    
    def select_action(self, state):
        """Epsilon-greedy action selection with random tie-breaking."""
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        
        return self.get_greedy_action(state)
    
    def start_episode(self, episode_num):
        """Called at the start of each episode."""
        self.current_episode = episode_num
    
    def get_greedy_action(self, state):
        """Return best action without exploration (for demo mode)."""
        q_values = self.q_table[state]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)
    
    def get_stats(self):
        """Return training statistics."""
        return {
            'states_explored': len(self.q_table),
            'epsilon': self.epsilon,
            'episode': self.current_episode
        }

    #---------Intrinsic Reward---------#
    def reset_episode(self):
        """Reset visit counters for new episode."""
        self.visit_counts.clear()
    
    def get_intrinsic_reward(self, state):
        """Calculate intrinsic reward: 1/sqrt(n(s))"""
        self.visit_counts[state] += 1
        return self.scale / math.sqrt(self.visit_counts[state])
