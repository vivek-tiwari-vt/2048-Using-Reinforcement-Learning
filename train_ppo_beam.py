import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
from agents.hybrid import PPOBeamHybridAgent
from environment.game_2048 import Game2048Env  # Fixed spelling of environment

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def train_agent(env, agent, num_episodes=5000, save_interval=100, eval_interval=50):
    """Train the PPO-Beam hybrid agent"""
    # Training metrics
    episode_rewards = []
    max_tiles = []
    evaluations = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        
        # Use different beam parameters based on training progress
        if episode > num_episodes * 0.8:  # Last 20% of training
            agent.increase_beam_influence(7, 4)  # Wider, deeper beam search
        elif episode > num_episodes * 0.5:  # Mid training
            agent.increase_beam_influence(5, 3)  # Default beam search
        
        # Episode loop
        while not done:
            # Get valid moves from environment
            valid_moves = env.get_valid_moves()
            
            # Select action using hybrid policy
            action, action_prob = agent.get_action(state, valid_moves)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience with reward shaping
            agent.remember(state, action, action_prob, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Update policy periodically during episode
            if len(agent.memory) >= agent.batch_size:
                agent.update()
        
        # Track metrics
        episode_rewards.append(total_reward)
        max_tile = np.max(state)
        max_tiles.append(max_tile)
        
        # Periodic evaluation
        if episode % eval_interval == 0:
            eval_score = evaluate_agent(env, agent, 5)
            evaluations.append(eval_score)
            print(f"Episode {episode}/{num_episodes} - Reward: {total_reward:.1f}, "
                 f"Max Tile: {max_tile}, Eval Score: {eval_score:.1f}")
        else:
            print(f"Episode {episode}/{num_episodes} - Reward: {total_reward:.1f}, Max Tile: {max_tile}")
            
        # Save model periodically
        if episode % save_interval == 0:
            agent.save(f"models/ppo_beam_ep{episode}.pt")
            
            # Plot progress
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(episode_rewards)
            plt.title("Episode Rewards")
            plt.subplot(1, 3, 2)
            plt.plot(max_tiles)
            plt.title("Max Tiles")
            plt.subplot(1, 3, 3)
            plt.plot(range(0, episode, eval_interval), evaluations)
            plt.title("Evaluation Scores")
            plt.savefig(f"models/ppo_beam_ep{episode}_progress.png")
            plt.close()
    
    return episode_rewards, max_tiles, evaluations

def evaluate_agent(env, agent, num_games=10):
    """Evaluate agent performance without exploration"""
    # Store original exploration rate
    original_rate = agent.exploration_rate
    agent.exploration_rate = 0.05  # Small exploration for evaluation
    
    scores = []
    for _ in range(num_games):
        state = env.reset()
        done = False
        game_score = 0
        
        while not done:
            valid_moves = env.get_valid_moves()
            action, _ = agent.get_action(state, valid_moves)
            next_state, reward, done, info = env.step(action)
            game_score += reward
            state = next_state
        
        scores.append(game_score)
    
    # Restore original exploration rate
    agent.exploration_rate = original_rate
    
    return np.mean(scores)

def visualize_gameplay(env, agent, delay=0.3):
    """Visualize agent gameplay"""
    state = env.reset()
    env.render()
    done = False
    total_reward = 0
    
    while not done:
        valid_moves = env.get_valid_moves()
        action, _ = agent.get_action(state, valid_moves)
        
        # Print action taken
        action_names = ["Left", "Up", "Right", "Down"]
        print(f"Action: {action_names[action]}")
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Display board
        env.render()
        state = next_state
        time.sleep(delay)
    
    print(f"Game over! Score: {total_reward}, Max tile: {np.max(state)}")
    return total_reward

if __name__ == "__main__":
    # Initialize 2048 environment
    env = Game2048Env()
    
    # Create directories for saving models and results
    os.makedirs("models", exist_ok=True)
    
    # Initialize PPO-Beam hybrid agent - fixed to work with any environment
    state_dim = 16  # For 2048 game (4x4 grid)
    action_dim = 4  # Four possible moves: left, up, right, down
    
    agent = PPOBeamHybridAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        beam_width=15,
        search_depth=30
    )
    
    # Set mode (train, eval, or visual)
    mode = "visual"  # Change to "visual" to see gameplay or "eval" to evaluate
    
    if mode == "train":
        # Train agent
        rewards, tiles, evals = train_agent(
            env, 
            agent, 
            num_episodes=3000,
            save_interval=100,
            eval_interval=50
        )
        
        # Final save
        agent.save("models/ppo_beam_final.pt")
        
    elif mode == "eval":
        # Load existing model if available
        if os.path.exists("models/ppo_beam_final.pt"):
            agent.load("models/ppo_beam_final.pt")
            
        # Evaluate agent performance
        avg_score = evaluate_agent(env, agent, num_games=20)
        print(f"Average score over 20 games: {avg_score}")
        
    elif mode == "visual":
        # Load existing model if available
        if os.path.exists("models/ppo_beam_final.pt"):
            agent.load("models/ppo_beam_final.pt")
        
        # Visualize gameplay
        visualize_gameplay(env, agent, delay=0.5)