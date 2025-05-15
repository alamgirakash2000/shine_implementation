import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',
                      help='Environment name (e.g., BreakoutNoFrameskip-v4, PongNoFrameskip-v4)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=100,
                      help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                      help='Enable rendering')
    return parser.parse_args()

class ShieldedAgent:
    def __init__(self, env_name, model_path):
        # Initialize environment
        self.env = gym.make(env_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load model
        self.model = PPO.load(model_path)
        
    def evaluate(self, num_episodes=100, render=False):
        rewards = []
        episode_lengths = []
        shield_activations = []
        
        for episode in tqdm(range(num_episodes)):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_shield_activations = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                    
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Check for shield activation (if implemented in the model)
                if hasattr(self.model, 'shield_activated') and self.model.shield_activated:
                    episode_shield_activations += 1
                    
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            shield_activations.append(episode_shield_activations)
            
        return rewards, episode_lengths, shield_activations

def plot_results(env_name, rewards, episode_lengths, shield_activations):
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards, label='Rewards')
    plt.title(f'{env_name} Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(episode_lengths, label='Episode Lengths')
    plt.title(f'{env_name} Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    
    # Plot shield activations
    if any(shield_activations):
        plt.subplot(2, 2, 3)
        plt.plot(shield_activations, label='Shield Activations')
        plt.title(f'{env_name} Shield Activations per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Activations')
        plt.legend()
        
        # Plot shield activation heatmap
        plt.subplot(2, 2, 4)
        plt.hist(shield_activations, bins=20)
        plt.title(f'{env_name} Shield Activation Distribution')
        plt.xlabel('Number of Activations')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{env_name.lower()}_shielded_agent_performance.png')
    plt.close()

def main():
    args = get_args()
    
    # Initialize shielded agent
    agent = ShieldedAgent(args.env, args.model_path)
    
    # Evaluate agent
    print(f"Evaluating {args.env} agent...")
    rewards, episode_lengths, shield_activations = agent.evaluate(
        num_episodes=args.episodes,
        render=args.render
    )
    
    # Plot results
    plot_results(args.env, rewards, episode_lengths, shield_activations)
    
    # Print statistics
    print(f"\n{args.env} Statistics:")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Success Rate: {(np.sum(np.array(rewards) > 0) / len(rewards)) * 100:.2f}%")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
    
    if any(shield_activations):
        print(f"Total Shield Activations: {sum(shield_activations)}")
        print(f"Average Shield Activations per Episode: {np.mean(shield_activations):.2f}")

if __name__ == "__main__":
    main() 