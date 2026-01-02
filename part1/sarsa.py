"""
SARSA implementation.
On-policy temporal difference learning.
"""
from base_agent import BaseAgent


class SARSA(BaseAgent):
    """SARSA agent - uses Q(s',a') where a' is the actual next action."""
    
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

    def update_intrinsic(self, state, action, env_reward, next_state, next_action, done):
        """SARSA update with combined reward."""
        intrinsic = self.get_intrinsic_reward(next_state)
        total_reward = env_reward + intrinsic
        self.update_normal(state, action, total_reward, next_state, next_action, done)
        return intrinsic