"""
Training loop for Q-Learning and SARSA.
"""
from gridworld import Gridworld
from q_learning import QLearning
from sarsa import SARSA
from intrinsic_reward import IntrinsicRewardWrapper
from visualization import TrainingStats
from config import TRAINING, MAX_STEPS_PER_EPISODE
import pygame
import os


class Trainer:
    """Handles training loop for RL algorithms."""
    
    def __init__(self, level, algorithm='q_learning', use_intrinsic=False, render=False):
        self.level = level
        self.algorithm = algorithm
        self.use_intrinsic = use_intrinsic
        self.render = render
        self.stats = TrainingStats()
        self.agent = None
    
    def create_agent(self, episodes):
        """Create the appropriate agent based on algorithm choice."""
        if self.algorithm == 'q_learning':
            base_agent = QLearning(episodes=episodes)
        else:
            base_agent = SARSA(episodes=episodes)
        
        if self.use_intrinsic:
            return IntrinsicRewardWrapper(base_agent)
        return base_agent
    
    def train(self, episodes=None, callback=None):
        """Run training loop for specified episodes."""
        episodes = episodes or TRAINING['episodes']
        self.agent = self.create_agent(episodes)
        env = Gridworld(self.level, render_mode=self.render)
        
        for episode in range(episodes):
            self.agent.start_episode(episode)
            state = env.reset()
            total_reward = 0
            done = False
            
            if self.algorithm == 'sarsa' and not self.use_intrinsic:
                action = self.agent.select_action(state)
            
            while not done:
                if self.algorithm == 'q_learning':
                    action = self.agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    
                    if self.use_intrinsic:
                        self.agent.update_q_learning(state, action, reward, next_state, done)
                    else:
                        self.agent.update(state, action, reward, next_state, done)
                        
                elif self.algorithm == 'sarsa':
                    next_state, reward, done = env.step(action)
                    next_action = self.agent.select_action(next_state)
                    self.agent.update(state, action, reward, next_state, next_action, done)
                    action = next_action
                
                total_reward += reward
                state = next_state
                
                if self.render:
                    env.render(f"Episode: {episode+1}/{episodes}")
                    if not env.handle_events():
                        env.close()
                        return self.stats
            
            # success = len(env.apples) == 0 and len(env.chests) == 0
            success = check_success(env)
            self.stats.record_episode(total_reward, env.steps, success)
            
            if callback:
                callback(episode, total_reward, success)
            
            if (episode + 1) % 100 == 0:
                avg_reward = sum(self.stats.episode_rewards[-100:]) / 100
                print(f"Episode {episode+1}/{episodes} - Avg Reward: {avg_reward:.2f}")
        
        env.close()
        return self.stats
    
    def demo(self, steps_per_second=5):
        """Run a visual demonstration of the learned policy."""
        if self.agent is None:
            print("No trained agent. Run train() first.")
            return
        
        env = Gridworld(self.level, render_mode=True)
        state = env.reset()
        running = True
        
        while running and not env.done:
            action = self.agent.get_greedy_action(state)
            state, reward, done = env.step(action)
            env.render("Demo Mode - Press ESC to exit")
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            
            env.clock.tick(steps_per_second)
        
        pygame.time.wait(1000)
        env.close()
    
    def visual_train(self, episodes=None, initial_speed=10):
        """Run training with visual display showing progress.
        
        Args:
            episodes: Number of episodes to train
            initial_speed: Initial steps per second (1-60)
        """
        
        episodes = episodes or TRAINING['episodes']
        self.agent = self.create_agent(episodes)
        env = Gridworld(self.level, render_mode=True)
        
        speed = initial_speed
        min_speed, max_speed = 1, 200
        running = True
        paused = False
        
        for episode in range(episodes):
            if not running:
                break
                
            self.agent.start_episode(episode)
            state = env.reset()
            total_reward = 0
            done = False
            
            if self.algorithm == 'sarsa' and not self.use_intrinsic:
                action = self.agent.select_action(state)
            
            while not done and running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_UP:
                            speed = min(speed + 5, max_speed)
                        elif event.key == pygame.K_DOWN:
                            speed = max(speed - 5, min_speed)
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                
                if paused:
                    env.render(f"Ep: {episode+1}/{episodes} | Reward: {total_reward:.0f} | Speed: {speed} | [PAUSED] Space to resume")
                    env.clock.tick(30)
                    continue
                
                # Execute step
                if self.algorithm == 'q_learning' or self.use_intrinsic:
                    action = self.agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    
                    if self.use_intrinsic:
                        self.agent.update_q_learning(state, action, reward, next_state, done)
                    else:
                        self.agent.update(state, action, reward, next_state, done)
                else:
                    next_state, reward, done = env.step(action)
                    next_action = self.agent.select_action(next_state)
                    self.agent.update(state, action, reward, next_state, next_action, done)
                    action = next_action
                
                total_reward += reward
                state = next_state
                
                # Render with info
                epsilon = getattr(self.agent, 'epsilon', 0)
                if hasattr(self.agent, 'base_agent'):
                    epsilon = self.agent.base_agent.epsilon
                info = f"Ep: {episode+1}/{episodes} | Reward: {total_reward:.0f} | ε: {epsilon:.2f} | Speed: {speed} (↑/↓)"
                env.render(info)
                env.clock.tick(speed)
            
            if running:
                success = len(env.apples) == 0 and len(env.chests) == 0
                self.stats.record_episode(total_reward, env.steps, success)
                
                if (episode + 1) % 50 == 0:
                    avg_reward = sum(self.stats.episode_rewards[-50:]) / min(50, len(self.stats.episode_rewards))
                    print(f"Episode {episode+1}/{episodes} - Avg Reward: {avg_reward:.2f}")
        
        env.close()
        return self.stats


def train_and_compare(level, episodes=500, save_dir="part1/results"):
    """Train both Q-Learning and SARSA, then compare results."""

    print(f"\n{'='*50}")
    print(f"Training on Level {level.level_id}: {level.name}")
    print(f"{'='*50}")
    
    print("\nTraining Q-Learning...")
    q_trainer = Trainer(level, algorithm='q_learning')
    q_stats = q_trainer.train(episodes)
    
    print("\nTraining SARSA...")
    s_trainer = Trainer(level, algorithm='sarsa')
    s_stats = s_trainer.train(episodes)
    
    from visualization import compare_training_curves
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"level_{level.level_id}_comparison.png")
    compare_training_curves(
        [q_stats, s_stats],
        ['Q-Learning', 'SARSA'],
        title=f"Level {level.level_id}: {level.name}",
        save_path=save_path
    )
    
    return q_trainer, s_trainer

def check_success(env) -> bool:
    return len(env.apples) == 0 and len(env.chests) == 0