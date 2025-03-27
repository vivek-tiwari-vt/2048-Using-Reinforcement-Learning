import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

def plot_learning_progress(rewards, highest_tiles, scores=None, save_path=None):
    """
    Plot the learning progress over episodes.
    
    Args:
        rewards: List of episode rewards
        highest_tiles: List of highest tiles reached in each episode
        scores: List of final scores for each episode
        save_path: Path to save the figure
    
    Returns:
        fig: Matplotlib figure object
    """
    # Determine number of subplots based on whether scores are provided
    if scores is not None and len(scores) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    episodes = range(1, len(rewards) + 1)
    
    # Plot rewards
    ax1 = axes[0]
    ax1.plot(episodes, rewards, 'b-', alpha=0.6, label='Episode Reward')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Calculate moving average if we have enough data
    if len(rewards) >= 10:
        window = min(10, len(rewards)//5)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(rewards) + 1), moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
        ax1.legend()
    
    # Plot highest tiles
    ax2 = axes[1]
    ax2.plot(episodes, highest_tiles, 'g-')
    ax2.set_title('Highest Tile Reached')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Tile Value')
    ax2.set_yscale('log', base=2)
    ax2.grid(True)
    
    # Set y-ticks to powers of 2
    max_tile = max(highest_tiles) if highest_tiles else 2
    yticks = [2**i for i in range(1, int(np.log2(max_tile)) + 2)]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([str(tick) for tick in yticks])
    
    # Plot scores if provided
    if scores is not None and len(scores) > 0:
        ax3 = axes[2]
        ax3.plot(episodes, scores, 'm-', alpha=0.6, label='Game Score')
        ax3.set_title('Game Scores')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Score')
        ax3.grid(True)
        
        # Calculate moving average if we have enough data
        if len(scores) >= 10:
            window = min(10, len(scores)//5)
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax3.plot(range(window, len(scores) + 1), moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
            ax3.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig

def visualize_board(board, title="2048 Game Board"):
    """
    Visualize the 2048 game board with colored tiles.
    
    Args:
        board: The game board as a 1D or 2D numpy array
        title: Title to display on the visualization
    
    Returns:
        fig: Matplotlib figure object
    """
    # Reshape the board if it's a 1D array
    if len(board.shape) == 1:
        # Assume it's a square board
        size = int(np.sqrt(len(board)))
        board = board.reshape(size, size)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Color mapping based on 2048 game colors
    colors = {
        0: '#CCC0B3',  # Empty tile
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
        4096: '#3C3A32'  # Beyond 2048
    }
    
    text_colors = {
        0: '#776E65',
        2: '#776E65',
        4: '#776E65',
        8: '#F9F6F2',
        16: '#F9F6F2',
        32: '#F9F6F2',
        64: '#F9F6F2',
        128: '#F9F6F2',
        256: '#F9F6F2',
        512: '#F9F6F2',
        1024: '#F9F6F2',
        2048: '#F9F6F2',
        4096: '#F9F6F2'
    }
    
    # Draw the board with a grid
    ax.set_facecolor('#BBADA0')  # Board background color
    
    # Draw each tile
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            value = int(board[i, j])
            
            # Get the color based on the value
            if value in colors:
                color = colors[value]
            else:
                color = '#3C3A32'  # Default for values beyond our color map
            
            # Get text color
            if value in text_colors:
                text_color = text_colors[value]
            else:
                text_color = '#F9F6F2'  # Default white text for high values
            
            # Draw tile with padding
            padding = 0.05
            rect = plt.Rectangle(
                (j + padding, (board.shape[0] - 1 - i) + padding),
                1 - 2*padding, 1 - 2*padding,
                facecolor=color, edgecolor='#BBADA0'
            )
            ax.add_patch(rect)
            
            # Add text for non-zero tiles
            if value != 0:
                fontsize = 24 if value < 100 else 20 if value < 1000 else 16
                ax.text(
                    j + 0.5, (board.shape[0] - 1 - i) + 0.5, str(value),
                    fontsize=fontsize, ha='center', va='center',
                    color=text_color, fontweight='bold'
                )
    
    # Set limits and remove ticks
    ax.set_xlim(0, board.shape[1])
    ax.set_ylim(0, board.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig