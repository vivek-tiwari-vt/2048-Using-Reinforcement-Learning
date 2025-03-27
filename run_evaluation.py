import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from environment.game_2048 import Game2048Env
from agents.beam_search_agent import BeamSearchAgent

def evaluate_beam_search(num_games=1000, beam_width=15, search_depth=30, 
                        render_freq=None, save_dir='beam_search_evaluation'):
    """
    Run beam search evaluation for many games and track statistics
    
    Args:
        num_games: Number of games to play
        beam_width: Beam width for the agent
        search_depth: Search depth for the agent
        render_freq: How often to render the game (None to disable)
        save_dir: Directory to save results
        
    Returns:
        Dict containing evaluation results
    """
    # Create timestamp for unique directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting evaluation of {num_games} games with beam_width={beam_width}, search_depth={search_depth}")
    print(f"Results will be saved to '{save_dir}'")
    
    # Initialize agent
    agent = BeamSearchAgent(beam_width=beam_width, search_depth=search_depth)
    
    # Initialize results tracking
    results = {
        'scores': [],
        'highest_tiles': [],
        'moves': [],
        'game_time': [],
        'final_boards': [],
        'best_board': None,
        'best_score': 0,
        'best_game_idx': 0
    }
    
    for game_idx in range(num_games):
        env = Game2048Env()
        state = env.reset()
        
        done = False
        moves = 0
        start_time = time.time()
        
        while not done:
            if render_freq and game_idx % render_freq == 0:
                env.render()
                time.sleep(0.1)
                
            # Get action from agent
            action, _ = agent.get_action(state)
            state, reward, done, info = env.step(action)
            moves += 1
            
            # Limit maximum moves to prevent infinite games
            if moves >= 5000:
                print(f"Game {game_idx+1} reached move limit")
                break
        
        # Record game results
        game_time = time.time() - start_time
        score = info['score']
        highest_tile = np.max(state)
        
        results['scores'].append(score)
        results['highest_tiles'].append(highest_tile)
        results['moves'].append(moves)
        results['game_time'].append(game_time)
        
        # Ensure we store the board in the right shape
        if len(state.shape) == 1:
            board_size = int(np.sqrt(state.shape[0]))
            results['final_boards'].append(state.copy().reshape(board_size, board_size))
        else:
            results['final_boards'].append(state.copy())
        
        # Update best game if this is the highest score
        if score > results['best_score']:
            results['best_score'] = score
            
            # Ensure the best board is stored in 2D format
            if len(state.shape) == 1:
                board_size = int(np.sqrt(state.shape[0]))
                results['best_board'] = state.copy().reshape(board_size, board_size)
            else:
                results['best_board'] = state.copy()
                
            results['best_game_idx'] = game_idx
            
        print(f"Game {game_idx+1}/{num_games}: Score = {score}, Highest Tile = {highest_tile}, Moves = {moves}")
        
        # Save intermediate results every 10 games
        if (game_idx + 1) % 10 == 0:
            create_visualizations(results, save_dir, beam_width, search_depth)
    
    # Create final visualizations
    create_visualizations(results, save_dir, beam_width, search_depth)
    
    # Print summary statistics
    print("\n==== EVALUATION SUMMARY ====")
    print(f"Highest tile reached: {max(results['highest_tiles'])}")
    print(f"Best score: {max(results['scores'])}")
    print(f"Average score: {sum(results['scores'])/len(results['scores']):.1f}")
    print(f"Average highest tile: {sum(results['highest_tiles'])/len(results['highest_tiles']):.1f}")
    
    # Check if 4096 was reached
    tile_counts = {}
    for tile in results['highest_tiles']:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    
    print("\nHighest tile distribution:")
    for tile in sorted(tile_counts.keys()):
        percentage = (tile_counts[tile] / num_games) * 100
        print(f"  {tile}: {tile_counts[tile]} games ({percentage:.1f}%)")
    
    if 4096 in tile_counts:
        print(f"\nReached 4096 tile in {tile_counts[4096]} games ({tile_counts[4096]/num_games*100:.1f}%)")
    
    return results

def create_visualizations(results, save_dir, beam_width, search_depth):
    """Create visualizations of the evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data for plotting
    scores = results['scores']
    highest_tiles = results['highest_tiles']
    moves = results['moves']
    
    # 1. Score Distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(scores, kde=True, bins=30)
    plt.title(f'Score Distribution (Beam Width={beam_width}, Search Depth={search_depth})')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'score_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Highest Tile Distribution
    plt.figure(figsize=(12, 8))
    # Convert to log2 for better visualization
    log2_tiles = [int(np.log2(tile)) for tile in highest_tiles]
    counts = {}
    for t in log2_tiles:
        counts[t] = counts.get(t, 0) + 1
    
    # Plot as a bar chart with actual tile values
    tiles = sorted(counts.keys())
    plt.bar([str(2**t) for t in tiles], [counts[t] for t in tiles], color='teal')
    plt.title(f'Highest Tile Distribution (Beam Width={beam_width}, Search Depth={search_depth})')
    plt.xlabel('Highest Tile Value')
    plt.ylabel('Number of Games')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'tile_distribution.png'), dpi=150)
    plt.close()
    
    # 3. Score vs. Highest Tile
    plt.figure(figsize=(12, 8))
    # Create dictionary mapping tile values to their average scores
    tile_scores = {}
    for tile, score in zip(highest_tiles, scores):
        if tile not in tile_scores:
            tile_scores[tile] = []
        tile_scores[tile].append(score)
    
    tile_avgs = {}
    for tile in tile_scores:
        tile_avgs[tile] = sum(tile_scores[tile]) / len(tile_scores[tile])
    
    # Sort by tile value and plot
    tiles = sorted(tile_avgs.keys())
    plt.bar([str(t) for t in tiles], [tile_avgs[t] for t in tiles], color='purple')
    plt.title(f'Average Score by Highest Tile (Beam Width={beam_width}, Search Depth={search_depth})')
    plt.xlabel('Highest Tile Value')
    plt.ylabel('Average Score')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'score_by_tile.png'), dpi=150)
    plt.close()
    
    # 4. Progress Plot (if enough games have been played)
    if len(scores) > 10:
        plt.figure(figsize=(14, 10))
        
        # Create moving averages
        window = min(50, len(scores)//10) if len(scores) >= 100 else 5
        
        # Score moving average
        score_ma = []
        for i in range(len(scores) - window + 1):
            score_ma.append(sum(scores[i:i+window]) / window)
        
        # Highest tile moving average (use log2 for readability)
        log2_tiles = [np.log2(t) for t in highest_tiles]
        tile_ma = []
        for i in range(len(log2_tiles) - window + 1):
            tile_ma.append(sum(log2_tiles[i:i+window]) / window)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Score plot
        ax1.plot(range(len(scores)), scores, 'b-', alpha=0.3, label='Score')
        if len(score_ma) > 0:
            ax1.plot(range(window-1, len(scores)), score_ma, 'r-', 
                    label=f'{window}-game Moving Avg')
        ax1.set_title(f'Score Progression (Beam Width={beam_width}, Search Depth={search_depth})')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Highest tile plot
        ax2.plot(range(len(highest_tiles)), log2_tiles, 'g-', alpha=0.3, label='Log2(Highest Tile)')
        if len(tile_ma) > 0:
            ax2.plot(range(window-1, len(log2_tiles)), tile_ma, 'r-', 
                    label=f'{window}-game Moving Avg')
        ax2.set_title('Highest Tile Progression')
        ax2.set_xlabel('Game Number')
        ax2.set_ylabel('Highest Tile (log₂)')
        
        # Set y-ticks to show actual tile values
        min_val = int(min(log2_tiles))
        max_val = int(max(log2_tiles)) + 1
        yticks = list(range(min_val, max_val))
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([f'{2**y} ({y})' for y in yticks])
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_over_time.png'), dpi=150)
        plt.close()
    
    # 5. Save Best Board Visualization
    if results['best_board'] is not None:
        visualize_board(results['best_board'], results['best_score'], save_dir)

def visualize_board(board, score, save_dir):
    """Create a visualization of the game board"""
    # Reshape the board if it's 1D
    if len(board.shape) == 1:
        board_size = int(np.sqrt(board.shape[0]))  # Should be 4 for a 2048 game
        board = board.reshape(board_size, board_size)
    
    # Create a figure with a specific size
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set up the grid
    for i in range(5):
        ax.axhline(i, color='black', linewidth=2)
        ax.axvline(i, color='black', linewidth=2)
    
    # Color mapping for tiles
    # Colors from original 2048 game (approximate)
    color_map = {
        0: '#CCC0B3',  # Empty cell
        2: '#EEE4DA',
        4: '#EDE0C8',
        8: '#F2B179',
        16: '#F59563',
        32: '#F67C5F',
        64: '#F65E3B',
        128: '#EDCF72',
        256: '#EDCC61',
        512: '#EDC850',
        1024: '#EDC53F',
        2048: '#EDC22E',
        4096: '#3C3A32',  # Dark color for 4096
    }
    
    # Fill each cell with the appropriate color and value
    for i in range(4):
        for j in range(4):
            value = int(board[i, j])
            color = color_map.get(value, '#3C3A32')  # Default dark color for very high values
            
            # Draw the colored rectangle
            rect = plt.Rectangle((j, 3-i), 1, 1, facecolor=color, edgecolor='gray', linewidth=2)
            ax.add_patch(rect)
            
            # Add the text
            if value != 0:
                # Adjust text size based on number of digits
                fontsize = 24 if value < 100 else 22 if value < 1000 else 18 if value < 10000 else 14
                plt.text(j + 0.5, 3-i + 0.5, str(value), 
                        fontsize=fontsize, ha='center', va='center', 
                        color='#776E65' if value < 8 else 'white', 
                        weight='bold')
    
    plt.title(f"Best Game Board (Score: {score})", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_board.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Beam Search Agent for 2048')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to play')
    parser.add_argument('--width', type=int, default=15, help='Beam width')
    parser.add_argument('--depth', type=int, default=30, help='Search depth')
    parser.add_argument('--render', type=int, default=None, help='Render frequency (None to disable)')
    parser.add_argument('--save-dir', type=str, default='beam_search_evaluation', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run the evaluation
    evaluate_beam_search(
        num_games=args.games,
        beam_width=args.width,
        search_depth=args.depth,
        render_freq=args.render,
        save_dir=args.save_dir
    )