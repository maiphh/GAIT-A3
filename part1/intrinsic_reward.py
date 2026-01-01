"""
Intrinsic reward extension for exploration bonus.
Wraps Q-Learning or SARSA with visit counting.
"""
import math
from collections import defaultdict
from config import INTRINSIC_SCALE


class IntrinsicRewardWrapper:
    """Adds intrinsic reward (1/sqrt(n(s))) to any RL agent."""
    
    def __init__(self, base_agent, scale=None):
        self.agent = base_agent
        self.scale = scale or INTRINSIC_SCALE
        self.visit_counts = defaultdict(int)
    
    def reset_episode(self):
        """Reset visit counters for new episode."""
        self.visit_counts.clear()
    
    def get_intrinsic_reward(self, state):
        """Calculate intrinsic reward: 1/sqrt(n(s))"""
        self.visit_counts[state] += 1
        return self.scale / math.sqrt(self.visit_counts[state])
    
    def select_action(self, state):
        return self.agent.select_action(state)
    
    def update_q_learning(self, state, action, env_reward, next_state, done):
        """Q-learning update with combined reward."""
        intrinsic = self.get_intrinsic_reward(state)
        total_reward = env_reward + intrinsic
        self.agent.update(state, action, total_reward, next_state, done)
        return intrinsic
    
    def update_sarsa(self, state, action, env_reward, next_state, next_action, done):
        """SARSA update with combined reward."""
        intrinsic = self.get_intrinsic_reward(state)
        total_reward = env_reward + intrinsic
        self.agent.update(state, action, total_reward, next_state, next_action, done)
        return intrinsic
    
    def start_episode(self, episode_num):
        self.reset_episode()
        self.agent.start_episode(episode_num)
    
    def get_greedy_action(self, state):
        return self.agent.get_greedy_action(state)
    
    def get_stats(self):
        stats = self.agent.get_stats()
        stats['unique_states_visited'] = len(self.visit_counts)
        return stats
    
    @property
    def q_table(self):
        return self.agent.q_table
