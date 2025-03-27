<<<<<<< HEAD
# 2048-Using-Reinforcement-Learning
=======
# 2048 Reinforcement Learning Project

## Overview

This project implements and evaluates various AI agents for playing the popular 2048 game using reinforcement learning and search techniques. The primary goal is to develop agents capable of achieving high scores and reaching the 2048 tile or beyond.

### Key Features

- **Multiple Agent Implementations**:
  - Proximal Policy Optimization (PPO) agent for learning-based approach
  - Beam Search agent for planning-based approach
  - Hybrid agent combining both techniques

- **Comprehensive Evaluation Framework**:
  - Performance metrics tracking (scores, highest tiles, move counts)
  - Visualization tools for game boards and learning progress
  - Detailed statistical analysis of agent performance

- **Flexible Training Pipeline**:
  - Configurable training parameters
  - Checkpoint saving and loading
  - Progress visualization during training

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Seaborn (for visualizations)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/2048_RL.git
   cd 2048_RL
   ```

2. Install the required dependencies:
   ```bash
   pip install torch numpy matplotlib seaborn pandas
   ```

## Usage

### Training an Agent

To train a PPO agent:

```bash
python main.py train --episodes 2000 --update-freq 10 --save-freq 100 --render-freq 100
```

Options:
- `--episodes`: Number of episodes to train for (default: 1000)
- `--max-steps`: Maximum steps per episode (default: 2000)
- `--update-freq`: Policy update frequency (default: 5)
- `--save-freq`: Checkpoint save frequency (default: 50)
- `--render-freq`: Game rendering frequency (default: 100)
- `--checkpoint-dir`: Directory to save checkpoints (default: 'checkpoints')

### Playing with a Trained Agent

To play the game using a trained agent:

```bash
python main.py play --model checkpoints/final_model.pth --delay 0.2 --visuals
```

Options:
- `--model`: Path to the trained model file
- `--max-steps`: Maximum steps to play (default: 2000)
- `--no-render`: Disable terminal rendering
- `--delay`: Delay between steps when rendering (default: 0.2)
- `--visuals`: Show matplotlib visualizations

### Using Beam Search Agent

To run the game with the beam search agent:

```bash
python main.py beam_search
```

### Evaluating Beam Search Performance

To run a comprehensive evaluation of the beam search agent:

```bash
python evaluate_beam_search.py --num-games 100 --beam-width 20 --search-depth 30
```

Options:
- `--num-games`: Number of games to evaluate (default: 100)
- `--beam-width`: Beam width for the agent (default: 15)
- `--search-depth`: Search depth for the agent (default: 20)
- `--render-freq`: How often to render games (default: None)

## Project Structure

```
2048_RL/
├── agents/                  # Agent implementations
│   ├── __init__.py
│   ├── beam_search_agent.py # Beam search agent implementation
│   ├── hybrid.py            # Hybrid agent (PPO + beam search)
│   └── ppo_agent.py         # PPO agent implementation
├── checkpoints/             # Saved model checkpoints
├── environment/             # Game environment
│   ├── __init__.py
│   └── game_2048.py         # 2048 game implementation
├── models/                  # Neural network models
│   ├── __init__.py
│   └── transformer.py       # Transformer model for PPO
├── results/                 # Evaluation results and visualizations
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── visualization.py     # Visualization tools
├── evaluate_beam_search.py  # Script for evaluating beam search agent
├── main.py                  # Main entry point
├── play.py                  # Script for playing with trained agents
├── run_evaluation.py        # Evaluation framework
├── train.py                 # Training framework
└── README.md                # This file
```

## Agent Descriptions

### PPO Agent

The PPO (Proximal Policy Optimization) agent uses deep reinforcement learning to learn a policy for playing 2048. It consists of:

- **Actor Network**: Determines action probabilities
- **Critic Network**: Estimates state values
- **Experience Replay**: Stores and samples game experiences
- **PPO Algorithm**: Updates policy with clipped objective function

The agent learns through self-play, gradually improving its policy through experience.

### Beam Search Agent

The Beam Search agent uses a planning-based approach with these key components:

- **Beam Width**: Number of candidate states to maintain (k)
- **Search Depth**: How deep to search in the game tree (d)
- **Evaluation Function**: Heuristics to evaluate board states including:
  - Monotonicity (tiles arranged in order)
  - Smoothness (adjacent tiles have similar values)
  - Free tiles (number of empty spaces)
  - Snake pattern (zigzag arrangement of decreasing values)

The agent achieved the 2048 tile in 35% of games and successfully reached the 4096 tile in evaluation.

### Hybrid Agent

The hybrid agent combines the strengths of both PPO and beam search approaches:

- Uses PPO for general policy learning
- Applies beam search for critical game states
- Dynamically switches between approaches based on game state

## Performance Results

### Agent Comparison

Comparative performance of all three agent implementations:

| Metric | PPO Agent | Beam Search Agent | Hybrid Agent |
|--------|-----------|-------------------|---------------|
| Highest Tile Achieved | 512 | 4096 | 512 |
| Best Score | ~5,000 | 51,372 | ~6,000 |
| Average Score | ~3,000 | 18,945.6 | ~3,500 |
| Average Highest Tile | ~256 | 1,315.8 | ~256 |
| Success Rate (512 tile) | 0.5% | N/A | 2% |
| Average Moves per Game | ~1,200 | ~1,500 | ~1,400 |

### Tile Achievement Rates

| Tile Value | PPO Agent | Beam Search Agent | Hybrid Agent |
|------------|-----------|-------------------|---------------|
| 256 | ~98% | 2% | ~95% |
| 512 | 2% | 14% | 5% |
| 1024 | 0% | 49% | 0% |
| 2048 | 0% | 34% | 0% |
| 4096 | 0% | 1% | 0% |

### Beam Search Agent Details

Based on evaluation of the beam search agent across 100 games:

| Metric | Value |
|--------|-------|
| Highest Tile Achieved | 4096 |
| Best Score | 51,372 |
| Average Score | 18,945.6 |
| Average Highest Tile | 1,315.8 |
| Success Rate (2048+ tiles) | 35% |

## Future Improvements

- Experiment with increased beam width for potentially higher 4096 success rates
- Implement a hybrid approach combining beam search with learning-based evaluation
- Add pattern-based endgame strategies for better handling of critical board states
- Optimize the evaluation functions for higher success rates at 4096

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original 2048 game by Gabriele Cirulli
- OpenAI for the PPO algorithm
- The reinforcement learning community for various techniques and insights
>>>>>>> 9bc3bf5 (Initial commit: Adding essential project files for 2048 RL implementation)
