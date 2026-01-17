"""
Q-Learning implementation.
Off-policy temporal difference learning.
"""
from base_agent import BaseAgent


class QLearning(BaseAgent):
    """Q-Learning agent - uses max Q(s',a') for updates."""
    
    def update(self, state, action, reward, next_state, done):
        if self.use_intrinsic:
            return self.update_intrinsic(state, action, reward, next_state, done)
        else:
            return self.update_normal(state, action, reward, next_state, done)

    def update_normal(self, state, action, reward, next_state, done):
        """Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]"""
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            max_next_q = max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q
        
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def update_intrinsic(self, state, action, env_reward, next_state, done):
        """Q-learning update with combined reward."""
        intrinsic = self.get_intrinsic_reward(next_state)
        total_reward = env_reward + intrinsic
        self.update_normal(state, action, total_reward, next_state, done)
        return intrinsic