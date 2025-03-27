import argparse
import os
from environment.game_2048 import Game2048Env
from agents.ppo_agent import PPOAgent
from agents.beam_search_agent import BeamSearchAgent
from train import train_agent
import numpy as np
import time

def main():
    parser = argparse.ArgumentParser(description='Train or play 2048 with AI')
    parser.add_argument('mode', choices=['train', 'play', 'evaluate', 'beam_search'], 
                        help='Mode to run the program')
    parser.add_argument('--episodes', type=int, default=1000, 
                        help='Number of episodes for training/evaluation')
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to model file for evaluation/play')
    parser.add_argument('--max-steps', type=int, default=2000, help='Maximum steps per episode/game')
    parser.add_argument('--update-freq', type=int, default=5, help='Policy update frequency')
    parser.add_argument('--save-freq', type=int, default=50, help='Checkpoint save frequency')
    parser.add_argument('--render-freq', type=int, default=100, help='Game rendering frequency (0 to disable)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--no-render', action='store_true', help='Disable terminal rendering')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between steps when rendering')
    parser.add_argument('--visuals', action='store_true', help='Show matplotlib visualizations')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(
            episodes=args.episodes,
            max_steps=args.max_steps,
            update_frequency=args.update_freq,
            save_frequency=args.save_freq,
            render_frequency=args.render_freq,
            checkpoint_dir=args.checkpoint_dir
        )
    
    elif args.mode == 'beam_search':
        # Run the game with beam search
        env = Game2048Env()
        agent = BeamSearchAgent(beam_width=10, search_depth=20)
        
        state = env.reset()
        env.render()
        time.sleep(0.5)
        
        done = False
        total_reward = 0
        moves = 0
        
        while not done:
            valid_moves = env.get_valid_moves()
            action, _ = agent.get_action(state, valid_moves)
            
            print(f"Move {moves}: {agent.action_names[action]}")
            state, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.3)
            
            total_reward += reward
            moves += 1
        
        print(f"Game over! Score: {info['score']}, Highest tile: {info['highest_tile']}")
        print(f"Total reward: {total_reward}, Moves: {moves}")

    elif args.mode == 'play':
        from play import play_game
        
        play_game(
            model_path=args.model,
            max_steps=args.max_steps,
            render=not args.no_render,
            render_delay=args.delay,
            visuals=args.visuals
        )

if __name__ == "__main__":
    main()