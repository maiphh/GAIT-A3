"""
SARSA implementation with epsilon-greedy policy.
On-policy temporal difference learning.
"""
import random
from collections import defaultdict
from config import TRAINING, NUM_ACTIONS, INTRINSIC_SCALE
import math


class SARSA:
    """SARSA agent with epsilon-greedy exploration."""
    
    def __init__(self, alpha=None, gamma=None, epsilon_start=None, epsilon_end=None, episodes=None, use_intrinsic=False, scale=INTRINSIC_SCALE):
        self.alpha = alpha or TRAINING['alpha']
        self.gamma = gamma or TRAINING['gamma']
        self.epsilon_start = epsilon_start or TRAINING['epsilon_start']
        self.epsilon_end = epsilon_end or TRAINING['epsilon_end']
        self.total_episodes = episodes or TRAINING['episodes']
        
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
    
    def update(self, state, action, reward, next_state, next_action, done):
        if self.use_intrinsic:
            return self.update_intrinsic(state, action, reward, next_state, next_action, done)
        else:
            return self.update_normal(state, action, reward, next_state, next_action, done)
    
    def update_normal(self, state, action, reward, next_state, next_action, done):
        """SARSA update: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]"""
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            next_q = self.q_table[next_state][next_action]
            target = reward + self.gamma * next_q
        
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)
    
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

    def update_intrinsic(self, state, action, env_reward, next_state, next_action, done):
        """SARSA update with combined reward."""
        intrinsic = self.get_intrinsic_reward(next_state)
        total_reward = env_reward + intrinsic
        self.update_normal(state, action, total_reward, next_state, next_action, done)
        return intrinsic
    