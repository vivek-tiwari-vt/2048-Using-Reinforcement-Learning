import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
from environment.game_2048 import Game2048Env
from agents.ppo_agent import PPOAgent
from utils.visualization import visualize_board

def play_game(model_path, max_steps=1000, render=True, render_delay=0.2, visuals=False):
    """
    Play a game of 2048 with a trained agent.
    
    Args:
        model_path: Path to the trained model
        max_steps: Maximum steps to play
        render: Whether to render the game in terminal
        render_delay: Delay between rendered steps
        visuals: Whether to show matplotlib visualizations
    
    Returns:
        score: Final score
        highest_tile: Highest tile achieved
    """
    # Initialize environment and agent
    env = Game2048Env()
    agent = PPOAgent()
    
    # Load model
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}. Using untrained agent.")
    
    # Start a new game
    state = env.reset()
    if render:
        print("Starting game...")
        env.render()
    
    total_reward = 0
    
    for step in range(max_steps):
        # Get action from agent
        action, _ = agent.get_action(state)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        state = next_state
        
        # Render if requested
        if render:
            print(f"Step {step}, Action: {['Left', 'Up', 'Right', 'Down'][action]}")
            env.render()
            if render_delay > 0:
                time.sleep(render_delay)
        
        # Show visual representation if requested
        if visuals and step % 5 == 0:
            fig = visualize_board(state, f"Step {step}, Score: {info['score']}")
            plt.draw()
            plt.pause(0.1)
            plt.close(fig)
        
        if done:
            break
    
    highest_tile = np.max(state)
    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Highest Tile: {highest_tile}")
    print(f"Steps: {step}")
    
    # Final visualization
    if visuals:
        fig = visualize_board(state, f"Final Board - Score: {info['score']}, Highest: {highest_tile}")
        plt.show()
    
    return info["score"], highest_tile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play 2048 with a trained agent')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='Path to the model file')
    parser.add_argument('--max-steps', type=int, default=2000, help='Maximum steps to play')
    parser.add_argument('--no-render', action='store_true', help='Disable terminal rendering')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between steps when rendering')
    parser.add_argument('--visuals', action='store_true', help='Show matplotlib visualizations')
    
    args = parser.parse_args()
    
    try:
        play_game(
            model_path=args.model,
            max_steps=args.max_steps,
            render=not args.no_render,
            render_delay=args.delay,
            visuals=args.visuals
        )
    except KeyboardInterrupt:
        print("\nGame stopped by user")