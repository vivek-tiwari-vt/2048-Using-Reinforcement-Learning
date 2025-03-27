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
    
    print(f"Starting training for {episodes} episodes with {agent_type}...")
    
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
            
            # Get action using valid moves if possible
            action, action_prob = agent.get_action(state, valid_moves)
            
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
            
            # Save agent configuration
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
            # Save agent configuration
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
                    agent.exploration_rate = min(0.8, agent.exploration_rate * 1.5)
                    print(f"Progress stalled, increasing exploration to {agent.exploration_rate}")
                else:
                    print("Progress stalled, but agent doesn't support exploration rate adjustment")
                stall_counter = 0
        else:
            stall_counter = 0
            last_best_tile = best_tile
    
    # Final save and progress plot
    agent.save(os.path.join(checkpoint_dir, f"{agent_type}_final_model.pth"))
    
    plot_learning_progress(episode_rewards, highest_tiles, game_scores,
                          os.path.join(checkpoint_dir, f"{agent_type}_final_progress.png"))
    
    # Print statistics
    print("\nTraining Complete!")
    print(f"Best Tile Achieved: {best_tile}")
    print(f"Final Score: {game_scores[-1]}")
    print(f"Valid Moves: {valid_moves_count}, Invalid Moves: {invalid_moves_count}")
    
    return agent, episode_rewards, highest_tiles, game_scores

def main():
    parser = argparse.ArgumentParser(description='Train a 2048 RL agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max-steps', type=int, default=2000, help='Maximum steps per episode')
    parser.add_argument('--update-freq', type=int, default=5, help='Policy update frequency')
    parser.add_argument('--save-freq', type=int, default=50, help='Checkpoint save frequency')
    parser.add_argument('--render-freq', type=int, default=100, help='Game rendering frequency (0 to disable)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Ask user which agent to use first
    print("\n=== 2048 AGENT SELECTION ===")
    print("Which agent would you like to train first?")
    print("1. PPO Agent (Deep Reinforcement Learning)")
    print("2. Beam Search Agent (Search-based Planning)")
    
    choice = ""
    while choice not in ["1", "2"]:
        choice = input("Enter your choice (1 or 2): ")
    
    # Initialize the selected agent
    ppo_agent = None
    beam_search_agent = None
    
    if choice == "1":
        print("\n=== CONFIGURING PPO AGENT ===")
        # Get PPO configuration options
        print("Using aggressive PPO Agent with very high exploration incentives")
        ppo_agent = PPOAgent()
        
        # Train PPO agent
        print("\n=== TRAINING PPO AGENT ===")
        train_agent(
            agent=ppo_agent,
            episodes=args.episodes,
            max_steps=args.max_steps,
            update_frequency=args.update_freq,
            save_frequency=args.save_freq,
            render_frequency=args.render_freq,
            checkpoint_dir=os.path.join(args.checkpoint_dir, "ppo"),
            debug=args.debug
        )
        
        # Ask if user wants to also try beam search
        print("\nDo you want to also train a Beam Search agent for comparison?")
        if input("Enter y/n: ").lower() == 'y':
            print("\n=== CONFIGURING BEAM SEARCH AGENT ===")
            beam_width = int(input("Enter beam width (recommended: 15-20): ") or 15)
            search_depth = int(input("Enter search depth (recommended: 20-30): ") or 25)
            beam_search_agent = BeamSearchAgent(beam_width=beam_width, search_depth=search_depth)
            
            print("\n=== TRAINING BEAM SEARCH AGENT ===")
            train_agent(
                agent=beam_search_agent,
                episodes=min(args.episodes, 200),  # Fewer episodes for beam search
                max_steps=args.max_steps,
                update_frequency=args.update_freq,
                save_frequency=args.save_freq,
                render_frequency=args.render_freq,
                checkpoint_dir=os.path.join(args.checkpoint_dir, "beam_search"),
                debug=args.debug
            )
    
    else:  # choice == "2"
        print("\n=== CONFIGURING BEAM SEARCH AGENT ===")
        beam_width = int(input("Enter beam width (recommended: 15-20): ") or 15)
        search_depth = int(input("Enter search depth (recommended: 20-30): ") or 25)
        beam_search_agent = BeamSearchAgent(beam_width=beam_width, search_depth=search_depth)
        
        # Train beam search agent
        print("\n=== TRAINING BEAM SEARCH AGENT ===")
        train_agent(
            agent=beam_search_agent,
            episodes=min(args.episodes, 200),  # Fewer episodes for beam search
            max_steps=args.max_steps,
            update_frequency=args.update_freq,
            save_frequency=args.save_freq,
            render_frequency=args.render_freq,
            checkpoint_dir=os.path.join(args.checkpoint_dir, "beam_search"),
            debug=args.debug
        )
        
        # Ask if user wants to also try PPO
        print("\nDo you want to also train a PPO agent for comparison?")
        if input("Enter y/n: ").lower() == 'y':
            print("\n=== CONFIGURING PPO AGENT ===")
            print("Using aggressive PPO Agent with very high exploration incentives")
            ppo_agent = PPOAgent()
            
            print("\n=== TRAINING PPO AGENT ===")
            train_agent(
                agent=ppo_agent,
                episodes=args.episodes,
                max_steps=args.max_steps,
                update_frequency=args.update_freq,
                save_frequency=args.save_freq,
                render_frequency=args.render_freq,
                checkpoint_dir=os.path.join(args.checkpoint_dir, "ppo"),
                debug=args.debug
            )
    
    # Compare results
    if ppo_agent is not None and beam_search_agent is not None:
        print("\n=== COMPARISON ===")
        print("Both agents have been trained. Check the visualization files in the checkpoints directory.")
        print("Consider using the evaluation script to compare their performance.")

if __name__ == "__main__":
    main()