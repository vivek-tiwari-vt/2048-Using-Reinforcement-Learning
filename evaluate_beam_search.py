import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
import datetime
import json
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from environment.game_2048 import Game2048Env
from agents.beam_search_agent import BeamSearchAgent

def run_game(agent, max_moves=10000, render=False, render_delay=0.1):
    """
    Run a single game with the provided agent
    
    Args:
        agent: The BeamSearchAgent to use
        max_moves: Maximum number of moves allowed
        render: Whether to render the game state
        render_delay: Delay between renders in seconds
        
    Returns:
        dict: Game results including score, highest tile, moves, etc.
    """
    env = Game2048Env()
    state = env.reset()
    
    if render:
        env.render()
        time.sleep(render_delay)
    
    done = False
    moves = 0
    valid_moves = 0
    invalid_moves = 0
    
    # Track when each milestone tile was first achieved
    milestones = {64: None, 128: None, 256: None, 512: None, 
                 1024: None, 2048: None, 4096: None, 8192: None}
    
    # Track board states for visualization
    board_history = [state.reshape(4, 4).copy()]
    
    # Track max tiles and scores over time for plotting
    max_tiles_history = [np.max(state)]
    scores_history = [0]  # Start with 0 score
    
    while not done and moves < max_moves:
        # Get the agent's action
        action, _ = agent.get_action(state)
        
        # Take the action
        prev_state = state.copy()
        state, reward, done, info = env.step(action)
        
        # Track milestone achievements
        max_tile = np.max(state)
        for milestone in milestones.keys():
            if max_tile >= milestone and milestones[milestone] is None:
                milestones[milestone] = moves
        
        # Track move validity
        if info['valid_move']:
            valid_moves += 1
        else:
            invalid_moves += 1
        
        # Visualization tracking
        board_history.append(state.reshape(4, 4).copy())
        max_tiles_history.append(max_tile)
        scores_history.append(info['score'])
        
        if render:
            env.render()
            if not np.array_equal(prev_state, state):  # Only delay for valid moves
                time.sleep(render_delay)
        
        moves += 1
    
    # Collect results
    results = {
        'score': info['score'],
        'highest_tile': int(np.max(state)),
        'moves': moves,
        'valid_moves': valid_moves,
        'invalid_moves': invalid_moves,
        'milestones': milestones,
        'board_history': board_history,
        'max_tiles_history': max_tiles_history,
        'scores_history': scores_history,
        'final_board': state.reshape(4, 4).copy()
    }
    
    return results

def run_evaluation(num_games=1000, beam_width=15, search_depth=20, 
                 render_freq=None, save_dir='results'):
    """
    Run evaluation of beam search agent on multiple games
    
    Args:
        num_games: Number of games to run
        beam_width: Width parameter for beam search
        search_depth: Depth parameter for beam search
        render_freq: If not None, render every nth game
        save_dir: Directory to save results and visualizations
        
    Returns:
        dict: Compiled results from all games
    """
    # Create timestamp for unique directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting evaluation of {num_games} games with beam_width={beam_width}, search_depth={search_depth}")
    print(f"Results will be saved to '{save_dir}'")
    
    # Initialize the agent
    agent = BeamSearchAgent(beam_width=beam_width, search_depth=search_depth)
    
    # Track all game results
    results = {
        'scores': [],
        'highest_tiles': [],
        'moves': [],
        'valid_moves': [],
        'invalid_moves': [],
        'milestones': {64: [], 128: [], 256: [], 512: [], 1024: [], 2048: [], 4096: [], 8192: []},
        'best_games': []  # Will store indices of games with highest scores/tiles
    }
    
    # Run all games
    for i in range(num_games):
        should_render = render_freq is not None and i % render_freq == 0
        game_result = run_game(agent, render=should_render)
        
        # Store results
        results['scores'].append(game_result['score'])
        results['highest_tiles'].append(game_result['highest_tile'])
        results['moves'].append(game_result['moves'])
        results['valid_moves'].append(game_result['valid_moves'])
        results['invalid_moves'].append(game_result['invalid_moves'])
        
        # Track milestones
        for milestone, move in game_result['milestones'].items():
            if move is not None:  # Only record achieved milestones
                results['milestones'][milestone].append(move)
        
        # Track best games (top 5 by score)
        if len(results['best_games']) < 5:
            results['best_games'].append(i)
            results['best_games'].sort(key=lambda idx: results['scores'][idx], reverse=True)
        elif game_result['score'] > results['scores'][results['best_games'][-1]]:
            results['best_games'][-1] = i
            results['best_games'].sort(key=lambda idx: results['scores'][idx], reverse=True)
        
        # Print progress
        if (i+1) % 10 == 0 or i == 0 or i == num_games - 1:
            print(f"Game {i+1}/{num_games}: Score = {game_result['score']}, "
                  f"Highest Tile = {game_result['highest_tile']}, "
                  f"Moves = {game_result['moves']}")
        
        # Save the best boards
        if game_result['highest_tile'] >= 2048:
            # Save this special board as an individual visualization
            plt.figure(figsize=(8, 8))
            visualize_board(game_result['final_board'], 
                           f"Game {i+1} - Score: {game_result['score']}, "
                           f"Highest Tile: {game_result['highest_tile']}")
            plt.savefig(os.path.join(save_dir, f"high_tile_game_{i+1}.png"), dpi=150)
            plt.close()
            
            # Also save a visualization of how this game evolved
            visualize_game_progression(game_result, 
                                      os.path.join(save_dir, f"progression_game_{i+1}.png"))
            
            # Save the full game data
            with open(os.path.join(save_dir, f"game_{i+1}_data.json"), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_result = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                      for k, v in game_result.items()}
                
                # Convert board history to list of lists
                serializable_result['board_history'] = [b.tolist() if isinstance(b, np.ndarray) else b 
                                                      for b in game_result['board_history']]
                serializable_result['final_board'] = serializable_result['final_board'].tolist()
                json.dump(serializable_result, f)
    
    # Create visualizations
    create_visualizations(results, save_dir, beam_width, search_depth)
    
    # Save overall results
    with open(os.path.join(save_dir, 'overall_results.json'), 'w') as f:
        # Convert numpy values to Python types
        json_results = {
            'scores': [int(score) for score in results['scores']],
            'highest_tiles': [int(tile) for tile in results['highest_tiles']],
            'moves': [int(move) for move in results['moves']],
            'valid_moves': [int(move) for move in results['valid_moves']],
            'invalid_moves': [int(move) for move in results['invalid_moves']],
            'milestones': {str(k): [int(m) for m in v] for k, v in results['milestones'].items()},
            'best_games': [int(idx) for idx in results['best_games']],
            'parameters': {
                'beam_width': beam_width,
                'search_depth': search_depth,
                'num_games': num_games
            }
        }
        json.dump(json_results, f, indent=4)
        
    print(f"\nEvaluation complete. Results saved to {save_dir}")
    return results

def visualize_board(board, title=None):
    """Visualize a 2048 board with pretty formatting"""
    plt.figure(figsize=(8, 8))
    
    # Define color map based on log2 of tile values
    cmap = plt.cm.YlOrRd  # Yellow -> Orange -> Red color map
    norm = LogNorm(vmin=2, vmax=8192)
    
    # Create background grid
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, 4, 1))
    ax.set_yticks(np.arange(-0.5, 4, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='gray', linestyle='-', linewidth=2)
    
    # Plot each tile
    for i in range(4):
        for j in range(4):
            value = board[i, j]
            if value > 0:
                # Color based on log2 of the value
                color = cmap(norm(value))
                text_color = 'white' if value >= 16 else 'black'  # Darker text on lighter tiles
                
                # Add the tile
                rect = plt.Rectangle((j-0.5, 3.5-i-0.5), 1, 1, facecolor=color, edgecolor='gray', linewidth=2)
                ax.add_patch(rect)
                
                # Add the number
                plt.text(j, 3-i, str(int(value)), fontsize=20, fontweight='bold', 
                        ha='center', va='center', color=text_color)
    
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 3.5)
    
    if title:
        plt.title(title, fontsize=16)
    
    return plt.gcf()

def visualize_game_progression(game_result, save_path=None):
    """Create visualization showing the progression of a game over time"""
    # Select key frames from the game (start, milestone achievements, end)
    board_history = game_result['board_history']
    max_tiles = game_result['max_tiles_history']
    scores = game_result['scores_history']
    milestones = game_result['milestones']
    
    # Get milestone frames
    key_frames = [0]  # Always include the start
    for milestone, move in sorted(milestones.items()):
        if move is not None:
            key_frames.append(move)
    
    # Include the end frame
    if key_frames[-1] != len(board_history) - 1:
        key_frames.append(len(board_history) - 1)
    
    # Limit to at most 8 frames for clarity
    if len(key_frames) > 8:
        # Always keep first and last, select others evenly
        first = key_frames[0]
        last = key_frames[-1]
        middle = np.array_split(key_frames[1:-1], 6)
        key_frames = [first] + [m[0] for m in middle] + [last]
    
    # Create the figure
    n_frames = len(key_frames)
    fig = plt.figure(figsize=(4 * n_frames, 12))
    
    # First row: board snapshots
    for i, frame_idx in enumerate(key_frames):
        ax = plt.subplot(3, n_frames, i+1)
        board = board_history[frame_idx]
        
        # Determine the title based on milestone achievements
        move_num = frame_idx
        title = f"Move {move_num}"
        
        # Check if this is a milestone achievement
        for milestone, move in milestones.items():
            if move == frame_idx and milestone >= 256:
                title += f"\n{milestone} Tile!"
        
        visualize_single_board(ax, board, title)
    
    # Second row: Max tile progression
    ax = plt.subplot(3, 1, 2)
    moves = range(len(max_tiles))
    ax.plot(moves, max_tiles, 'b-', linewidth=2)
    ax.set_ylabel('Max Tile', fontsize=14)
    ax.set_xlabel('Move Number', fontsize=14)
    ax.set_title('Maximum Tile Progression', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Mark milestone achievements
    for milestone, move in milestones.items():
        if move is not None and milestone >= 128:
            ax.plot(move, milestone, 'ro', markersize=8)
            ax.annotate(str(milestone), (move, milestone), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold')
    
    # Use log scale for y-axis
    ax.set_yscale('log', base=2)
    yticks = [2**i for i in range(1, int(np.log2(np.max(max_tiles)))+2)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(tick) for tick in yticks])
    
    # Third row: Score progression
    ax = plt.subplot(3, 1, 3)
    ax.plot(moves, scores, 'g-', linewidth=2)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Move Number', fontsize=14)
    ax.set_title('Score Progression', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    summary = (f"Final Score: {scores[-1]}\n"
               f"Highest Tile: {max_tiles[-1]}\n"
               f"Total Moves: {len(max_tiles)-1}\n"
               f"Valid Moves: {game_result['valid_moves']}")
    
    plt.figtext(0.02, 0.02, summary, fontsize=12, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig

def visualize_single_board(ax, board, title=None):
    """Visualize a single board on the provided axes"""
    # Define color map based on log2 of tile values
    cmap = plt.cm.YlOrRd  # Yellow -> Orange -> Red color map
    norm = LogNorm(vmin=2, vmax=8192)
    
    # Create background grid
    ax.set_xticks(np.arange(-0.5, 4, 1))
    ax.set_yticks(np.arange(-0.5, 4, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='gray', linestyle='-', linewidth=2)
    
    # Plot each tile
    for i in range(4):
        for j in range(4):
            value = board[i, j]
            if value > 0:
                # Color based on log2 of the value
                color = cmap(norm(value))
                text_color = 'white' if value >= 16 else 'black'  # Darker text on lighter tiles
                
                # Add the tile
                rect = plt.Rectangle((j-0.5, 3.5-i-0.5), 1, 1, facecolor=color, edgecolor='gray', linewidth=2)
                ax.add_patch(rect)
                
                # Add the number
                ax.text(j, 3-i, str(int(value)), fontsize=14, fontweight='bold', 
                      ha='center', va='center', color=text_color)
    
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    
    if title:
        ax.set_title(title, fontsize=12)

def create_visualizations(results, save_dir, beam_width, search_depth):
    """Create comprehensive visualizations from the results"""
    # Set a consistent style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results['scores'], kde=True, bins=20, color='blue')
    plt.title(f'Score Distribution (Beam Width={beam_width}, Depth={search_depth})', fontsize=16)
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    avg_score = np.mean(results['scores'])
    median_score = np.median(results['scores'])
    max_score = np.max(results['scores'])
    
    stats_text = (f"Average: {avg_score:.1f}\n"
                  f"Median: {median_score:.1f}\n"
                  f"Maximum: {max_score:.1f}")
    
    plt.figtext(0.7, 0.7, stats_text, fontsize=12, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.savefig(os.path.join(save_dir, 'score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Highest Tile Distribution
    plt.figure(figsize=(10, 6))
    # Convert to log2 for better visualization
    log2_tiles = [int(np.log2(tile)) for tile in results['highest_tiles']]
    unique_tiles = sorted(set(results['highest_tiles']))
    
    counts = {}
    for tile in unique_tiles:
        counts[tile] = results['highest_tiles'].count(tile)
    
    # Plot as a horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        [str(tile) for tile in unique_tiles],
        [counts[tile] for tile in unique_tiles],
        color=plt.cm.viridis(np.linspace(0, 1, len(unique_tiles)))
    )
    
    # Add percentage labels
    for bar, tile in zip(bars, unique_tiles):
        count = counts[tile]
        percentage = 100 * count / len(results['highest_tiles'])
        plt.text(
            bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{count} ({percentage:.1f}%)",
            va='center', fontsize=10
        )
    
    plt.title(f'Highest Tile Distribution (Beam Width={beam_width}, Depth={search_depth})', fontsize=16)
    plt.xlabel('Number of Games', fontsize=14)
    plt.ylabel('Highest Tile Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add stats
    plt.figtext(
        0.7, 0.2,
        f"Total Games: {len(results['highest_tiles'])}\n"
        f"Games with 2048+: {sum(1 for t in results['highest_tiles'] if t >= 2048)}\n"
        f"Games with 4096+: {sum(1 for t in results['highest_tiles'] if t >= 4096)}",
        fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'highest_tile_distribution.png'), dpi=150)
    plt.close()
    
    # 3. Score vs. Highest Tile Scatter Plot
    plt.figure(figsize=(12, 8))
    tile_jitter = np.random.normal(0, 0.05, len(results['highest_tiles']))
    
    # Create a colormap based on move count
    norm = plt.Normalize(min(results['moves']), max(results['moves']))
    colors = plt.cm.viridis(norm(results['moves']))
    
    # Create scatter plot
    plt.scatter(
        np.array(results['highest_tiles']) + tile_jitter,
        results['scores'],
        c=results['moves'],
        alpha=0.7,
        cmap='viridis',
        edgecolors='black',
        linewidths=0.5
    )
    
    # Set log scale for x-axis
    plt.xscale('log', base=2)
    xticks = [2**i for i in range(1, int(np.log2(max(results['highest_tiles'])))+1)]
    plt.xticks(xticks, [str(tick) for tick in xticks])
    
    plt.title(f'Score vs. Highest Tile (Beam Width={beam_width}, Depth={search_depth})', fontsize=16)
    plt.xlabel('Highest Tile Achieved', fontsize=14)
    plt.ylabel('Final Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Number of Moves', fontsize=12)
    
    # Add trend line
    z = np.polyfit(np.log2(results['highest_tiles']), results['scores'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(np.log2(results['highest_tiles'])), max(np.log2(results['highest_tiles'])), 100)
    plt.plot(2**x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'score_vs_highest_tile.png'), dpi=150)
    plt.close()
    
    # 4. Heat Map of Tile Positions
    # First, we need to collect tile position data from the best games
    best_boards = []
    for idx in results['best_games']:
        # Load the game data
        try:
            with open(os.path.join(save_dir, f"game_{idx+1}_data.json"), 'r') as f:
                game_data = json.load(f)
                best_boards.append(game_data['final_board'])
        except (FileNotFoundError, KeyError):
            # Skip if we can't load this game data
            continue
    
    if best_boards:
        # Create heatmaps showing where different tile values appear
        plt.figure(figsize=(15, 10))
        
        # Look at different tile values of interest
        tile_values = [64, 128, 256, 512, 1024, 2048]
        num_tiles = len(tile_values)
        
        for i, tile_value in enumerate(tile_values):
            ax = plt.subplot(2, 3, i+1)
            heatmap_data = np.zeros((4, 4))
            
            # Count occurrences of this tile value at each position
            for board in best_boards:
                for row in range(4):
                    for col in range(4):
                        if board[row][col] == tile_value:
                            heatmap_data[row][col] += 1
            
            # Plot the heatmap
            sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", cbar=False, ax=ax)
            ax.set_title(f"{tile_value} Tile Positions", fontsize=14)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tile_position_heatmap.png'), dpi=150)
        plt.close()
    
    # 5. Milestone Achievement Chart
    plt.figure(figsize=(12, 8))
    milestone_values = [64, 128, 256, 512, 1024, 2048, 4096]
    milestone_colors = plt.cm.viridis(np.linspace(0, 1, len(milestone_values)))
    
    for i, milestone in enumerate(milestone_values):
        # Skip if no games achieved this milestone
        if not results['milestones'][milestone]:
            continue
            
        # Calculate the percentage of games that achieved this milestone
        achievement_rate = len(results['milestones'][milestone]) / len(results['scores']) * 100
        
        # Calculate average move when this milestone was reached
        avg_move = np.mean(results['milestones'][milestone])
        
        plt.bar(i, achievement_rate, color=milestone_colors[i], alpha=0.7)
        plt.text(i, achievement_rate + 2, f"{achievement_rate:.1f}%\n({avg_move:.0f} moves)", 
                ha='center', fontweight='bold', fontsize=10)
    
    plt.title(f'Milestone Achievement Rates (Beam Width={beam_width}, Depth={search_depth})', fontsize=16)
    plt.xlabel('Milestone Tile Value', fontsize=14)
    plt.ylabel('Percentage of Games Achieving Milestone', fontsize=14)
    plt.xticks(range(len(milestone_values)), [str(m) for m in milestone_values])
    plt.ylim(0, 105)  # Leave room for the percentage labels
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'milestone_achievement_rates.png'), dpi=150)
    plt.close()
    
    # 6. Performance Over Time (Games) - Smoothed Line Chart
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Top subplot: Scores
    ax1 = plt.subplot(gs[0])
    games = range(1, len(results['scores'])+1)
    
    # Plot individual points
    ax1.scatter(games, results['scores'], alpha=0.3, color='blue', s=10)
    
    # Plot a smoothed line (moving average)
    window_size = min(50, len(results['scores'])//10)
    if window_size > 1:
        smoothed_scores = np.convolve(results['scores'], np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size, len(results['scores'])+1), smoothed_scores, 
                color='darkblue', linewidth=2, label=f'{window_size}-Game Moving Average')
    
    ax1.set_ylabel('Score', fontsize=14)
    ax1.set_title(f'Performance Over Games (Beam Width={beam_width}, Depth={search_depth})', fontsize=16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: Highest tiles
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # Convert to log2 scale for better visualization
    log2_tiles = np.log2(results['highest_tiles'])
    
    # Plot individual points
    ax2.scatter(games, log2_tiles, alpha=0.3, color='green', s=10)
    
    # Plot a smoothed line
    if window_size > 1:
        smoothed_tiles = np.convolve(log2_tiles, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size, len(log2_tiles)+1), smoothed_tiles, 
                color='darkgreen', linewidth=2, label=f'{window_size}-Game Moving Average')
    
    ax2.set_xlabel('Game Number', fontsize=14)
    ax2.set_ylabel('Highest Tile (log₂)', fontsize=14)
    
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
    
    # 7. Moves Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results['moves'], kde=True, bins=20, color='green')
    plt.title(f'Moves Distribution (Beam Width={beam_width}, Depth={search_depth})', fontsize=16)
    plt.xlabel('Number of Moves', fontsize=14)
    plt.ylabel('Frequency# filepath: /Volumes/DATA/2048_RL/evaluate_beam_search.py')