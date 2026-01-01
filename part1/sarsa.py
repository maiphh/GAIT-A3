"""
SARSA implementation with epsilon-greedy policy.
On-policy temporal difference learning.
"""
import random
from collections import defaultdict
from config import TRAINING, NUM_ACTIONS


class SARSA:
    """SARSA agent with epsilon-greedy exploration."""
    
    def __init__(self, alpha=None, gamma=None, epsilon_start=None, epsilon_end=None, episodes=None):
        self.alpha = alpha or TRAINING['alpha']
        self.gamma = gamma or TRAINING['gamma']
        self.epsilon_start = epsilon_start or TRAINING['epsilon_start']
        self.epsilon_end = epsilon_end or TRAINING['epsilon_end']
        self.total_episodes = episodes or TRAINING['episodes']
        
        self.q_table = defaultdict(lambda: [0.0] * NUM_ACTIONS)
        self.current_episode = 0
    
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
