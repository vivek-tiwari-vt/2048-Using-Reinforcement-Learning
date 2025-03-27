import numpy as np
import os
import time
import argparse
from environment.game_2048 import Game2048Env
from agents.ppo_agent import PPOAgent
from agents.beam_search_agent import BeamSearchAgent
from utils.visualization import plot_learning_progress, visualize_board
import matplotlib.pyplot as plt

def train_agent(agent, episodes=2000, max_steps=2000, update_frequency=10, 
                save_frequency=100, render_frequency=100, checkpoint_dir='checkpoints',
                debug=False):
    """
    Train a 2048 agent using either PPO or beam search.

    Args:
        agent: The agent to train (PPO or beam search)
        episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        update_frequency: How often to update the policy
        save_frequency: How often to save checkpoints
        render_frequency: How often to render the game
        checkpoint_dir: Directory to save checkpoints
        debug: Whether to print debug information
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    env = Game2048Env()
    agent_type = agent.__class__.__name__
    
    best_tile = 0  # Track the best highest tile reached so far
    episode_rewards = []
    highest_tiles = []
    game_scores = []  # Track game scores
    valid_moves_count = 0
    invalid_moves_count = 0
    
    print(f"Starting training for {episodes} episodes...")
    
    # Print device info for PPO agent
    if hasattr(agent, 'device'):
        print(f"Using device: {agent.device}")
    
    stall_counter = 0
    last_best_tile = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        moveset = []  # Track the sequence of moves for this episode
        episode_valid_moves = 0
        episode_invalid_moves = 0
        
        for step in range(max_steps):
            # Get valid moves
            valid_moves = env.get_valid_moves()
            
            try:
                # Use valid_moves if the agent supports it
                action, action_prob = agent.get_action(state, valid_moves)
            except TypeError:
                # Fallback if the agent doesn't support valid_moves
                action, action_prob = agent.get_action(state)
            
            # Record the move
            moveset.append(action)
            
            if debug:
                print(f"Step {step}: Chosen Action: {['Left', 'Up', 'Right', 'Down'][action]}")
            
            # Take action
            prev_state = state.copy()
            next_state, reward, done, info = env.step(action)
            
            # Track valid/invalid moves
            if info["valid_move"]:
                episode_valid_moves += 1
            else:
                episode_invalid_moves += 1
                
            if debug and np.array_equal(prev_state, next_state):
                print("Invalid Move! Board did not change.")
                print(f"Reward: {reward}, Score: {info['score']}, Highest Tile: {np.max(next_state)}")

            # Store experience (only for PPO agent)
            if hasattr(agent, 'remember'):
                agent.remember(state, action, action_prob, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Render if requested
            if render_frequency > 0 and episode % render_frequency == 0 and step % 50 == 0:
                env.render()
                time.sleep(0.05)
            
            # Update policy periodically (only for PPO agent)
            if hasattr(agent, 'update') and step % update_frequency == 0:
                try:
                    agent.update()
                except Exception as e:
                    if debug:
                        print(f"Update failed: {e}")
            
            if done:
                break
        
        # Final update at the end of the episode (only for PPO agent)
        if hasattr(agent, 'update'):
            try:
                agent.update()
            except Exception as e:
                if debug:
                    print(f"Final update failed: {e}")
        
        # Track episode performance
        episode_rewards.append(episode_reward)
        highest_tile = info.get("highest_tile", np.max(state))
        highest_tiles.append(highest_tile)
        game_scores.append(info.get("score", 0))
        
        # Update total valid/invalid moves count
        valid_moves_count += episode_valid_moves
        invalid_moves_count += episode_invalid_moves
        
        # Save the model if a new best tile is reached
        if highest_tile > best_tile:
            best_tile = highest_tile
            
            # Only save model weights for PPO agent
            if hasattr(agent, 'save'):
                agent.save(os.path.join(checkpoint_dir, f"{agent_type}_best_model_tile_{highest_tile}.pth"))
            
            # Save a visualization of the best board
            fig = visualize_board(state, f"Best Board (Score: {info['score']}, Tile: {highest_tile})")
            fig.savefig(os.path.join(checkpoint_dir, f"{agent_type}_best_board_tile_{highest_tile}.png"), dpi=150)
            plt.close(fig)
            
            # Save the moveset that achieved this tile
            with open(os.path.join(checkpoint_dir, f"{agent_type}_best_moveset_tile_{highest_tile}.txt"), "w") as f:
                f.write(",".join(map(str, moveset)))
        
        # Periodic checkpoint saving
        if episode % save_frequency == 0 and episode > 0:
            # Only save model weights for PPO agent
            if hasattr(agent, 'save'):
                agent.save(os.path.join(checkpoint_dir, f"{agent_type}_model_episode_{episode}.pth"))
            
            # Create and save progress visualization
            plot_learning_progress(episode_rewards, highest_tiles, game_scores,
                                  os.path.join(checkpoint_dir, f"{agent_type}_progress_episode_{episode}.png"))
        
        print(f"Episode {episode}: Score = {info['score']}, Highest Tile = {highest_tile}, "
              f"Valid Moves = {episode_valid_moves}, Invalid Moves = {episode_invalid_moves}")
        
        # Early stopping if the 2048 tile is reached
        if highest_tile >= 2048:
            print(f"Solved in {episode} episodes!")
            break
        
        # Detect stalling in progress
        if episode > 100 and best_tile == last_best_tile:
            stall_counter += 1
            if stall_counter >= 50:  # No improvement for 50 episodes
                # For PPO agent, try to encourage exploration
                if hasattr(agent, 'exploration_rate'):
                    agent.exploration_rate = min(0.4, getattr(agent, 'exploration_rate', 0.1) * 1.5)
                    print(f"Progress stalled, increasing exploration to {agent.exploration_rate}")
                else:
                    print("Progress stalled, but agent doesn't support exploration rate adjustment")
                stall_counter = 0
        else:
            stall_counter = 0
            last_best_tile = best_tile
    
    # Final save and progress plot
    if hasattr(agent, 'save'):
        agent.save(os.path.join(checkpoint_dir, f"{agent_type}_final_model.pth"))
    
    plot_learning_progress(episode_rewards, highest_tiles, game_scores,
                          os.path.join(checkpoint_dir, f"{agent_type}_final_progress.png"))
    
    # Print statistics
    print("\nTraining Complete!")
    print(f"Best Tile Achieved: {best_tile}")
    print(f"Final Score: {game_scores[-1]}")
    print(f"Valid Moves: {valid_moves_count}, Invalid Moves: {invalid_moves_count}")
    
    return agent, episode_rewards, highest_tiles, game_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a 2048 RL agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max-steps', type=int, default=2000, help='Maximum steps per episode')
    parser.add_argument('--update-freq', type=int, default=5, help='Policy update frequency')
    parser.add_argument('--save-freq', type=int, default=50, help='Checkpoint save frequency')
    parser.add_argument('--render-freq', type=int, default=100, help='Game rendering frequency (0 to disable)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Ask user which agent to use
    print("Select an agent type:")
    print("1. PPO Agent")
    print("2. Beam Search Agent")
    
    choice = ""
    while choice not in ["1", "2"]:
        choice = input("Enter your choice (1 or 2): ")
    
    # Initialize the selected agent
    if choice == "1":
        agent = PPOAgent()
        print("Using PPO Agent")
    else:
        beam_width = int(input("Enter beam width (recommended: 10-20): ") or 15)
        search_depth = int(input("Enter search depth (recommended: 20-40): ") or 30)
        agent = BeamSearchAgent(beam_width=beam_width, search_depth=search_depth)
        print(f"Using Beam Search Agent with width={beam_width}, depth={search_depth}")
    
    train_agent(
        agent=agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        update_frequency=args.update_freq,
        save_frequency=args.save_freq,
        render_frequency=args.render_freq,
        checkpoint_dir=args.checkpoint_dir,
        debug=args.debug
    )
