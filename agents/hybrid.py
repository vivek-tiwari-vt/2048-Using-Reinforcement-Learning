# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
# from collections import deque
# import matplotlib.pyplot as plt
# import os
# import sys
# import importlib.util

# # Dynamically import the Game2048Env
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# game_path = os.path.join(project_root, 'environment', 'game_2048.py')
# if os.path.exists(game_path):
#     spec = importlib.util.spec_from_file_location("game_2048", game_path)
#     game_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(game_module)
#     Game2048Env = game_module.Game2048Env
# else:
#     raise FileNotFoundError(f"Could not find game_2048.py at {game_path}")

# # Hybrid CNN-Transformer Model
# class HybridDQN(nn.Module):
#     def __init__(self, input_dim=16, output_dim=4):
#         super(HybridDQN, self).__init__()
#         # CNN for local feature extraction
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
#             nn.ReLU()
#         )
#         # Transformer for global context
#         # Fix the input dimension - CNN outputs (64, 4, 4) = 1024 features
#         self.embedding = nn.Linear(64 * 4 * 4, 128)  # Changed from 64*3*3
#         encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         # Final projection
#         self.fc = nn.Linear(128, output_dim)
    
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.view(-1, 1, 4, 4)  # Reshape to 4x4 grid
#         x = self.cnn(x)
#         # Print shapes for debugging (can remove after confirming)
#         # print(f"CNN output shape: {x.shape}")
#         x = x.view(batch_size, -1)  # Flatten CNN features
#         # print(f"Flattened shape: {x.shape}")
#         x = self.embedding(x)
#         x = x.unsqueeze(1)  # Add sequence dimension
#         x = self.transformer(x)
#         x = x.squeeze(1)
#         return self.fc(x)

# # Enhanced Replay Buffer with prioritization
# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, alpha=0.6):
#         self.buffer = deque(maxlen=capacity)
#         self.priorities = deque(maxlen=capacity)
#         self.alpha = alpha
    
#     def push(self, state, action, reward, next_state, done):
#         # Add experience with max priority for new experiences
#         max_priority = max(self.priorities, default=1.0)
#         self.buffer.append((state, action, reward, next_state, done))
#         self.priorities.append(max_priority)
    
#     def sample(self, batch_size, beta=0.4):
#         if len(self.buffer) < batch_size:
#             return random.sample(self.buffer, len(self.buffer)), range(len(self.buffer)), np.ones(len(self.buffer))
        
#         priorities = np.array(self.priorities, dtype=np.float32)
#         probs = priorities ** self.alpha
#         probs /= probs.sum()
        
#         indices = np.random.choice(len(self.buffer), batch_size, p=probs)
#         samples = [self.buffer[idx] for idx in indices]
        
#         # Importance sampling weights
#         weights = (len(self.buffer) * probs[indices]) ** (-beta)
#         weights /= weights.max()
        
#         return samples, indices, weights
    
#     def update_priorities(self, indices, priorities):
#         for idx, priority in zip(indices, priorities):
#             if idx < len(self.priorities):
#                 self.priorities[idx] = max(priority, 1e-5)  # Avoid zero priority
    
#     def __len__(self):
#         return len(self.buffer)

# # Advanced DQN Agent with hybrid model and PER
# class DQNAgent:
#     def __init__(self, env, model, target_model, replay_buffer, batch_size=256, gamma=0.99,
#                  epsilon_start=1.0, epsilon_end=0.001, decay_steps=150000, target_update_freq=250, learning_rate=1e-3):
#         self.env = env
#         self.model = model
#         self.target_model = target_model
#         self.replay_buffer = replay_buffer
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.epsilon = epsilon_start
#         self.epsilon_start = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.decay_steps = decay_steps
#         self.target_update_freq = target_update_freq
#         self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#         self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=decay_steps, eta_min=learning_rate/10)
#         self.step_counter = 0
#         self.device = torch.device("cuda" if torch.cuda.is_available() else 
#                                  "mps" if torch.backends.mps.is_available() else "cpu")
#         self.model.to(self.device)
#         self.target_model.to(self.device)
#         self.update_target_model()
        
#         # Exponential tile weighting
#         self.tile_weights = {2**i: 2**(i-1) for i in range(1, 16)}
        
#         # Snake pattern (proven effective for 2048)
#         self.snake_pattern = np.array([
#             [15, 14, 13, 12],
#             [8, 9, 10, 11],
#             [7, 6, 5, 4],
#             [0, 1, 2, 3]
#         ])
        
#         print(f"Using device: {self.device}")

#     def update_target_model(self):
#         self.target_model.load_state_dict(self.model.state_dict())

#     def select_action(self, state):
#         # Strategic exploration-exploitation
#         if random.random() < self.epsilon:
#             # Even during exploration, use strategic biasing
#             board = state.reshape(4, 4)
#             valid_moves = self.env.get_valid_moves()
#             valid_actions = [i for i, valid in enumerate(valid_moves) if valid]
            
#             if not valid_actions:  # If no valid moves, take random action (shouldn't happen)
#                 return random.randint(0, 3)
            
#             # Strategic bias based on board state
#             max_tile = np.max(board)
#             max_pos = np.unravel_index(np.argmax(board), (4, 4))
#             move_prefs = np.ones(4)
            
#             # If max tile is in bottom-right (ideal for snake pattern)
#             if max_tile >= 64 and max_pos == (3, 3):
#                 move_prefs[2] *= 3  # Right bias
#                 move_prefs[3] *= 3  # Down bias
            
#             # Filter valid actions and normalize preferences
#             valid_prefs = [move_prefs[a] for a in valid_actions]
#             total = sum(valid_prefs)
#             valid_prefs = [p/total for p in valid_prefs]
            
#             return random.choices(valid_actions, weights=valid_prefs)[0]
#         else:
#             # Exploit: use model
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
#             with torch.no_grad():
#                 q_values = self.model(state_tensor).cpu().numpy()[0]
            
#             # Filter valid moves
#             valid_moves = self.env.get_valid_moves()
            
#             # Apply large negative bias to invalid moves
#             for i in range(4):
#                 if not valid_moves[i]:
#                     q_values[i] = -1e9
                    
#             return np.argmax(q_values)

#     def train_step(self):
#         if len(self.replay_buffer) < self.batch_size:
#             return
            
#         # Sample with prioritization
#         batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=0.4 + 0.6 * min(self.step_counter / self.decay_steps, 1.0))
#         states, actions, rewards, next_states, dones = zip(*batch)
        
#         # Convert to tensors
#         states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
#         actions = torch.tensor(actions, dtype=torch.long).to(self.device)
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
#         next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
#         dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
#         weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

#         # Advanced reward shaping
#         shaped_rewards = []
#         with torch.no_grad():
#             for i, ns in enumerate(next_states.cpu().numpy()):
#                 board = ns.reshape(4, 4)
#                 prev_board = states[i].cpu().numpy().reshape(4, 4)
                
#                 # Base reward from environment
#                 base_reward = rewards[i].item()
#                 shaped_reward = base_reward * 0.1  # Reduce environment reward weight
                
#                 # 1. Exponential scaling for high tiles
#                 max_tile = np.max(board)
#                 if max_tile > 0:
#                     shaped_reward += np.log2(max_tile) * 2.0
                
#                 # 2. Super aggressive snake pattern enforcement
#                 snake_score = 0
#                 for i in range(4):
#                     for j in range(4):
#                         if board[i, j] > 0:
#                             # Heavier weighting for higher tiles in snake pattern
#                             tile_value = np.log2(board[i, j])
#                             position_value = 16 - self.snake_pattern[i, j]
#                             snake_score += tile_value * position_value
                
#                 # Normalize and apply with high weight
#                 shaped_reward += (snake_score / 500) * 10.0
                
#                 # 3. Massive corner bonus (critical for 2048 strategy)
#                 if max_tile > 64:
#                     if board[3, 3] == max_tile:  # Bottom-right corner (best)
#                         shaped_reward += np.log2(max_tile) * 5.0
#                     elif board[0, 0] == max_tile:  # Top-left corner (second best)
#                         shaped_reward += np.log2(max_tile) * 2.0
                
#                 # 4. Empty cell bonus (crucial for movement)
#                 empty_count = np.sum(board == 0)
#                 shaped_reward += empty_count * 0.5
                
#                 # 5. Merge potential bonus
#                 merge_bonus = 0
#                 # Check horizontal merges
#                 for i in range(4):
#                     for j in range(3):
#                         if board[i, j] > 0 and board[i, j] == board[i, j+1]:
#                             merge_bonus += board[i, j]
                
#                 # Check vertical merges
#                 for j in range(4):
#                     for i in range(3):
#                         if board[i, j] > 0 and board[i, j] == board[i+1, j]:
#                             merge_bonus += board[i, j]
                
#                 shaped_reward += merge_bonus * 0.01
                
#                 # 6. Huge bonus for new max tile
#                 if max_tile > np.max(prev_board):
#                     shaped_reward += max_tile * 0.5
                
#                 # Save the shaped reward
#                 shaped_rewards.append(shaped_reward)
        
#         shaped_rewards = torch.tensor(shaped_rewards, dtype=torch.float32).to(self.device)
        
#         # Q-learning update with Double DQN
#         # Current Q-values
#         current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
#         # Next actions from current policy
#         with torch.no_grad():
#             next_actions = self.model(next_states).argmax(1, keepdim=True)
#             # Q-values of next_actions from target network
#             next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
#             # Target Q-values
#             target_q_values = shaped_rewards + (1 - dones) * self.gamma * next_q_values
        
#         # Huber loss with prioritization weights
#         loss_fn = nn.SmoothL1Loss(reduction='none')
#         td_errors = loss_fn(current_q_values, target_q_values)
#         weighted_loss = (weights * td_errors).mean()
        
#         # Optimization
#         self.optimizer.zero_grad()
#         weighted_loss.backward()
#         # Gradient clipping for stability
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
#         self.optimizer.step()
#         self.scheduler.step()
        
#         # Update priorities in replay buffer
#         with torch.no_grad():
#             priorities = (td_errors + 1e-5).detach().cpu().numpy()  # Small constant for stability
#             self.replay_buffer.update_priorities(indices, priorities)
        
#         # Epsilon decay (faster early on, slower later)
#         self.step_counter += 1
#         progress = min(self.step_counter / self.decay_steps, 1.0)
#         self.epsilon = max(self.epsilon_end, 
#                           self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (progress**0.6))
        
#         # Target network update
#         if self.step_counter % self.target_update_freq == 0:
#             self.update_target_model()

# # Training and Evaluation Functions
# def train_agent(agent, env, num_episodes=10000, max_steps=2000):
#     """Train the DQN agent"""
#     print(f"Starting training for {num_episodes} episodes...")
    
#     # Tracking metrics
#     all_rewards = []
#     all_max_tiles = []
#     epsilon_values = []
    
#     best_max_tile = 0
#     best_episode_score = 0
    
#     # Preload buffer with a few random experiences to avoid empty buffer issues
#     print("Initializing experience buffer with random actions...")
#     for _ in range(agent.batch_size * 2):
#         state = env.reset()
#         action = random.randint(0, 3)
#         next_state, reward, done, _ = env.step(action)
#         agent.replay_buffer.push(state, action, reward, next_state, done)
#     print(f"Buffer initialized with {len(agent.replay_buffer)} experiences")
    
#     for episode in range(1, num_episodes + 1):
#         if episode == 1:
#             print(f"Starting episode {episode}...")
            
#         state = env.reset()
#         total_reward = 0
#         step = 0
#         done = False
        
#         # Episode loop
#         while not done and step < max_steps:
#             if episode == 1 and step % 100 == 0:
#                 print(f"  Episode 1, Step {step}")
                
#             step += 1
            
#             # Get action from agent
#             action = agent.select_action(state)
            
#             # Take action in environment
#             next_state, reward, done, info = env.step(action)
            
#             # Store transition in replay buffer
#             agent.replay_buffer.push(state, action, reward, next_state, done)
            
#             # Update agent (with timeout protection)
#             if episode == 1 and step == 1:
#                 print("  Running first training step...")
                
#             agent.train_step()
            
#             if episode == 1 and step == 1:
#                 print("  Completed first training step successfully")
                
#             state = next_state
#             total_reward += reward
        
#         # First episode completion message  
#         if episode == 1:
#             print(f"Episode 1 completed successfully in {step} steps")
            
#         # Track metrics
#         max_tile = np.max(state)
#         all_rewards.append(total_reward)
#         all_max_tiles.append(max_tile)
#         epsilon_values.append(agent.epsilon)
        
#         # Update records
#         if max_tile > best_max_tile:
#             best_max_tile = max_tile
#         if total_reward > best_episode_score:
#             best_episode_score = total_reward
        
#         # Progress reporting
#         if episode % 50 == 0:
#             avg_reward = np.mean(all_rewards[-50:])
#             avg_max_tile = np.mean(all_max_tiles[-50:])
#             print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.1f} | "
#                   f"Avg Max Tile: {avg_max_tile:.1f} | Best Tile: {best_max_tile} | "
#                   f"Epsilon: {agent.epsilon:.3f}")
            
#             # Quick evaluation every 500 episodes
#             if episode % 500 == 0:
#                 eval_score, eval_tile = evaluate_agent(agent, env, num_games=5)
#                 print(f"EVAL | Score: {eval_score:.1f} | Max Tile: {eval_tile}")
                
#                 # Save checkpoint for good performance
#                 if eval_tile >= 1024:
#                     save_model(agent.model, f"results/model_checkpoint_{episode}_{int(eval_tile)}.pt")
                    
#         # Save model periodically
#         if episode % 1000 == 0:
#             save_model(agent.model, f"results/model_episode_{episode}.pt")
            
#     print(f"Training completed. Best max tile: {best_max_tile}, Best episode score: {best_episode_score:.1f}")
#     return all_rewards, all_max_tiles, epsilon_values

# def evaluate_agent(agent, env, num_games=10, render=False):
#     """Evaluate the agent's performance without exploration"""
#     saved_epsilon = agent.epsilon
#     agent.epsilon = 0.01  # Minimal exploration during evaluation
    
#     total_scores = []
#     total_max_tiles = []
    
#     for game in range(num_games):
#         state = env.reset()
#         done = False
#         total_score = 0
        
#         while not done:
#             action = agent.select_action(state)
#             next_state, reward, done, info = env.step(action)
#             total_score += reward
#             state = next_state
            
#             if render:
#                 env.render()
#                 print(f"Action: {env.ACTIONS[action]}, Reward: {reward}")
#                 time.sleep(0.1)
                
#         max_tile = np.max(state)
#         total_scores.append(total_score)
#         total_max_tiles.append(max_tile)
        
#         if render:
#             print(f"Game {game+1} - Score: {total_score}, Max Tile: {max_tile}")
    
#     # Restore exploration rate
#     agent.epsilon = saved_epsilon
    
#     return np.mean(total_scores), np.max(total_max_tiles)

# def save_model(model, path):
#     """Save model weights"""
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     torch.save(model.state_dict(), path)
#     print(f"Model saved to {path}")

# # Visualization Functions
# def plot_scores(scores, save_path="results/scores.png"):
#     """Plot scores over episodes"""
#     plt.figure(figsize=(10, 5))
#     plt.plot(scores)
#     plt.title("Score per Episode")
#     plt.xlabel("Episode")
#     plt.ylabel("Score")
    
#     # Add rolling average
#     if len(scores) > 100:
#         rolling_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
#         plt.plot(rolling_avg, color='red', label="100-episode average")
#         plt.legend()
    
#     plt.savefig(save_path)
#     plt.close()

# def plot_highest_tiles(highest_tiles, save_path="results/highest_tiles.png"):
#     """Plot highest tile reached in each episode"""
#     plt.figure(figsize=(10, 5))
#     plt.plot(highest_tiles)
#     plt.title("Max Tile per Episode")
#     plt.xlabel("Episode")
#     plt.ylabel("Max Tile")
#     plt.savefig(save_path)
#     plt.close()

# def plot_tile_frequency(highest_tiles, save_path="results/tile_frequency.png"):
#     """Plot frequency of highest tile values"""
#     plt.figure(figsize=(12, 6))
#     unique, counts = np.unique(highest_tiles, return_counts=True)
    
#     # Convert to log scale for better visualization
#     log_values = [int(np.log2(v)) if v > 0 else 0 for v in unique]
#     labels = [f"2^{v}={int(2**v)}" for v in log_values]
    
#     plt.bar(range(len(counts)), counts, tick_label=labels)
#     plt.title("Frequency of Max Tile Values")
#     plt.xlabel("Tile Value")
#     plt.ylabel("Frequency")
#     plt.xticks(rotation=45)
#     plt.savefig(save_path)
#     plt.close()

# def plot_epsilon_decay(epsilon_values, save_path="results/epsilon_decay.png"):
#     """Plot epsilon decay over episodes"""
#     plt.figure(figsize=(10, 5))
#     plt.plot(epsilon_values)
#     plt.title("Epsilon Decay Over Episodes")
#     plt.xlabel("Episode")
#     plt.ylabel("Epsilon")
#     plt.savefig(save_path)
#     plt.close()

# # Main function
# def main():
#     env = Game2048Env()
    
#     # Create results folder if it doesn't exist
#     results_folder = "results"
#     os.makedirs(results_folder, exist_ok=True)
    
#     # Initialize model, target model and replay buffer
#     model = HybridDQN()
#     target_model = HybridDQN()
#     replay_buffer = PrioritizedReplayBuffer(capacity=200000)
    
#     # Start with a smaller batch size for stability
#     initial_batch_size = 32  # Start smaller for first 500 episodes
#     agent = DQNAgent(
#         env=env,
#         model=model,
#         target_model=target_model,
#         replay_buffer=replay_buffer,
#         batch_size=initial_batch_size,  # Smaller initial batch size
#         gamma=0.99,
#         epsilon_start=1.0,
#         epsilon_end=0.001,
#         decay_steps=150000,
#         target_update_freq=250,
#         learning_rate=1e-3
#     )
    
#     # Train the agent
#     print("Starting new training run with aggressive hybrid CNN-Transformer model.")
#     print(f"Using initial batch size: {initial_batch_size}")
#     scores, highest_tiles, epsilon_values = train_agent(agent, env, num_episodes=10000, max_steps=2000)
    
#     # Save final model
#     final_model_path = os.path.join(results_folder, "hybrid_cnn_transformer_final.pt")
#     save_model(agent.model, final_model_path)
    
#     # Plot results
#     plot_scores(scores)
#     plot_highest_tiles(highest_tiles)
#     plot_tile_frequency(highest_tiles)
#     plot_epsilon_decay(epsilon_values)
    
#     # Final evaluation
#     print("\nFinal evaluation:")
#     eval_score, eval_tile = evaluate_agent(agent, env, num_games=20)
#     print(f"Average Score: {eval_score:.1f}, Best Tile: {eval_tile}")
    
#     print(f"\nTraining completed. Results saved to {results_folder}")
#     print(f"Highest Score: {max(scores):.1f}")
#     print(f"Highest Tile: {max(highest_tiles)}")

# if __name__ == "__main__":
#     import time
#     main()




import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import sys
import importlib.util
import time

# Dynamically import the Game2048Env
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
game_path = os.path.join(project_root, 'environment', 'game_2048.py')
if os.path.exists(game_path):
    spec = importlib.util.spec_from_file_location("game_2048", game_path)
    game_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(game_module)
    Game2048Env = game_module.Game2048Env
else:
    raise FileNotFoundError(f"Could not find game_2048.py at {game_path}")

# Add simulation methods to Game2048Env
def simulate_move(self, board, action):
    """Simulate a move without changing game state, returns list of (next_state, reward, done)"""
    # Make a deep copy of the board
    board_copy = np.copy(board)
    
    # Save original state for comparison
    original_board = np.copy(board_copy)
    
    # Simulate the move
    if action == 0:  # Left
        board_copy = self._simulate_move_left(board_copy)
    elif action == 1:  # Up
        board_copy = np.rot90(board_copy)
        board_copy = self._simulate_move_left(board_copy)
        board_copy = np.rot90(board_copy, k=3)
    elif action == 2:  # Right
        board_copy = np.fliplr(board_copy)
        board_copy = self._simulate_move_left(board_copy)
        board_copy = np.fliplr(board_copy)
    elif action == 3:  # Down
        board_copy = np.rot90(board_copy, k=3)
        board_copy = self._simulate_move_left(board_copy)
        board_copy = np.rot90(board_copy)
    
    # Check if the move changed the board
    valid_move = not np.array_equal(original_board, board_copy)
    
    # If invalid move, return early with penalty
    if not valid_move:
        return [(board_copy, -1.0, False)]
    
    # Find empty cells for new tile placement
    empty_cells = list(zip(*np.where(board_copy == 0)))
    if not empty_cells:
        # No empty cells, game might be over
        return [(board_copy, 0.0, True)]
    
    # For beam search, consider multiple possible tile placements
    possible_states = []
    
    # Sample a few positions (limit for efficiency)
    sample_size = min(3, len(empty_cells))
    sampled_cells = random.sample(empty_cells, sample_size)
    
    for pos in sampled_cells:
        # 2 tile (90% probability)
        board_with_2 = np.copy(board_copy)
        board_with_2[pos] = 2
        reward_2 = self._calculate_simulation_reward(board_with_2, original_board)
        possible_states.append((board_with_2, reward_2 * 0.9, False))  # Weighted by 90% probability
        
        # 4 tile (10% probability)
        board_with_4 = np.copy(board_copy)
        board_with_4[pos] = 4
        reward_4 = self._calculate_simulation_reward(board_with_4, original_board)
        possible_states.append((board_with_4, reward_4 * 0.1, False))  # Weighted by 10% probability
    
    return possible_states

def _simulate_move_left(self, board):
    """Simulate left move on a copy of the board"""
    result = np.copy(board)
    rows, cols = board.shape
    
    for i in range(rows):
        # Extract non-zero elements
        row = result[i]
        non_zero = row[row != 0]
        
        if len(non_zero) == 0:
            continue
        
        # New row after merging
        new_row = []
        skip_next = False
        
        # Process each tile for merging
        for j in range(len(non_zero)):
            if skip_next:
                skip_next = False
                continue
            
            # Check if we can merge with the next tile
            if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                merged_value = non_zero[j] * 2
                new_row.append(merged_value)
                skip_next = True
            else:
                new_row.append(non_zero[j])
        
        # Pad with zeros
        new_row = np.array(new_row + [0] * (cols - len(new_row)))
        result[i] = new_row
    
    return result

def _calculate_simulation_reward(self, new_board, old_board):
    """Calculate reward for a simulated move"""
    # Basic reward: score increase from merges
    old_sum = np.sum(old_board)
    new_sum = np.sum(new_board)
    merge_reward = new_sum - old_sum
    
    # Bonus for higher max tile
    old_max = np.max(old_board)
    new_max = np.max(new_board)
    max_tile_bonus = 0
    if new_max > old_max:
        max_tile_bonus = new_max
    
    # Bonus for empty cells (crucial for movement)
    empty_bonus = np.sum(new_board == 0) * 0.1
    
    # Return combined reward
    return merge_reward + max_tile_bonus + empty_bonus

# Add the simulation methods to Game2048Env
Game2048Env.simulate_move = simulate_move
Game2048Env._simulate_move_left = _simulate_move_left
Game2048Env._calculate_simulation_reward = _calculate_simulation_reward

# Hybrid CNN-Transformer Model
class HybridDQN(nn.Module):
    def __init__(self, input_dim=16, output_dim=4):
        super(HybridDQN, self).__init__()
        # CNN for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
        )
        # Transformer for global context
        # Fix the input dimension - CNN outputs (64, 4, 4) = 1024 features
        self.embedding = nn.Linear(64 * 4 * 4, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Final projection
        self.fc = nn.Linear(128, output_dim)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, 1, 4, 4)  # Reshape to 4x4 grid
        x = self.cnn(x)
        x = x.view(batch_size, -1)  # Flatten CNN features
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

# Enhanced Replay Buffer with prioritization
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
    
    def push(self, state, action, reward, next_state, done):
        # Add experience with max priority for new experiences
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer)), range(len(self.buffer)), np.ones(len(self.buffer))
        
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = max(priority, 1e-5)  # Avoid zero priority
    
    def __len__(self):
        return len(self.buffer)

# Advanced DQN Agent with hybrid model, PER, and beam search
class DQNAgent:
    def __init__(self, env, model, target_model, replay_buffer, batch_size=256, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.001, decay_steps=150000, target_update_freq=250, learning_rate=1e-3):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.target_update_freq = target_update_freq
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=decay_steps, eta_min=learning_rate/10)
        self.step_counter = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.update_target_model()
        
        # Beam search parameters
        self.beam_width = 15
        self.search_depth = 30
        self.use_beam_search = True
        self.beam_search_threshold = 64  # Only use beam search for boards with tiles >= 64
        
        # Exponential tile weighting
        self.tile_weights = {2**i: 2**(i-1) for i in range(1, 16)}
        
        # Snake pattern (proven effective for 2048)
        self.snake_pattern = np.array([
            [15, 14, 13, 12],
            [8, 9, 10, 11],
            [7, 6, 5, 4],
            [0, 1, 2, 3]
        ])
        
        print(f"Using device: {self.device}")
        print(f"Beam search settings: width={self.beam_width}, depth={self.search_depth}")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def beam_search(self, state):
        """Use beam search for planning optimal moves sequence"""
        # Get board state
        board = state.reshape(4, 4)
        max_tile = np.max(board)
        
        # Only use beam search for complex board states (efficiency optimization)
        if max_tile < self.beam_search_threshold or np.sum(board > 0) < 8:
            # For simpler boards, use Q-network directly
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            # Filter valid moves
            valid_moves = self.env.get_valid_moves()
            for i in range(4):
                if not valid_moves[i]:
                    q_values[i] = -1e9
            
            return np.argmax(q_values)
        
        # Initialize beam with current state
        beam = [(board, [], 0, 1.0)]  # (state, action_sequence, cumulative_reward, probability)
        
        # Beam search loop for each step in the search depth
        for step in range(self.search_depth):
            candidates = []
            
            # For each state in the beam
            for current_board, actions, cum_reward, probability in beam:
                # Try all possible actions
                for action in range(4):
                    # Simulate the action and its possible outcomes
                    transitions = self.env.simulate_move(current_board, action)
                    
                    for next_board, reward, done in transitions:
                        # For leaf nodes, estimate value using neural network
                        if step == self.search_depth - 1 or done:
                            state_tensor = torch.tensor(next_board.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                value_estimate = self.model(state_tensor).max().item()
                            total_reward = cum_reward + reward + self.gamma * value_estimate * (1 - done)
                        else:
                            total_reward = cum_reward + reward
                        
                        # Calculate transition probability (for simplicity, equal probability among transitions)
                        new_probability = probability / len(transitions)
                        
                        # Store the candidate with its action sequence
                        new_actions = actions + [action]
                        candidates.append((next_board, new_actions, total_reward, new_probability))
            
            # Keep only top-k candidates for the next iteration
            candidates.sort(key=lambda x: x[2] * x[3], reverse=True)  # Sort by expected reward (reward * probability)
            beam = candidates[:self.beam_width]
            
            # If all candidates lead to game over, break early
            if all(done for _, _, _, done in beam):
                break
        
        # Extract first actions from the best sequences
        if not beam:
            # Fallback to random valid move
            valid_moves = self.env.get_valid_moves()
            valid_indices = [i for i, valid in enumerate(valid_moves) if valid]
            return random.choice(valid_indices if valid_indices else range(4))
        
        # Count how often each initial action appears in the beam, weighted by expected reward
        action_scores = {}
        for _, actions, reward, prob in beam:
            if actions:  # Check if action sequence is not empty
                first_action = actions[0]
                score = reward * prob
                if first_action in action_scores:
                    action_scores[first_action] += score
                else:
                    action_scores[first_action] = score
        
        # Choose best initial action
        if action_scores:
            return max(action_scores, key=action_scores.get)
        else:
            # Fallback to model prediction
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            # Filter valid moves
            valid_moves = self.env.get_valid_moves()
            for i in range(4):
                if not valid_moves[i]:
                    q_values[i] = -1e9
            
            return np.argmax(q_values)

    def select_action(self, state):
        """Select action using epsilon-greedy with beam search for exploitation"""
        # Exploration
        if random.random() < self.epsilon:
            # Strategic exploration (existing code)
            board = state.reshape(4, 4)
            valid_moves = self.env.get_valid_moves()
            valid_actions = [i for i, valid in enumerate(valid_moves) if valid]
            
            if not valid_actions:
                return random.randint(0, 3)
            
            # Strategic bias based on board state
            max_tile = np.max(board)
            max_pos = np.unravel_index(np.argmax(board), (4, 4))
            move_prefs = np.ones(4)
            
            # If max tile is in bottom-right (ideal for snake pattern)
            if max_tile >= 64 and max_pos == (3, 3):
                move_prefs[2] *= 3  # Right bias
                move_prefs[3] *= 3  # Down bias
            
            # Filter valid actions and normalize preferences
            valid_prefs = [move_prefs[a] for a in valid_actions]
            total = sum(valid_prefs)
            valid_prefs = [p/total for p in valid_prefs]
            
            return random.choices(valid_actions, weights=valid_prefs)[0]
        else:
            # Exploitation: use beam search for planning when appropriate
            if self.use_beam_search and np.max(state.reshape(4,4)) >= self.beam_search_threshold:
                return self.beam_search(state)
            else:
                # Standard model prediction
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.model(state_tensor).cpu().numpy()[0]
                
                # Filter valid moves
                valid_moves = self.env.get_valid_moves()
                for i in range(4):
                    if not valid_moves[i]:
                        q_values[i] = -1e9
                        
                return np.argmax(q_values)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample with prioritization
        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=0.4 + 0.6 * min(self.step_counter / self.decay_steps, 1.0))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Advanced reward shaping
        shaped_rewards = []
        with torch.no_grad():
            for i, ns in enumerate(next_states.cpu().numpy()):
                board = ns.reshape(4, 4)
                prev_board = states[i].cpu().numpy().reshape(4, 4)
                
                # Base reward from environment
                base_reward = rewards[i].item()
                shaped_reward = base_reward * 0.1  # Reduce environment reward weight
                
                # 1. Exponential scaling for high tiles
                max_tile = np.max(board)
                if max_tile > 0:
                    shaped_reward += np.log2(max_tile) * 2.0
                
                # 2. Super aggressive snake pattern enforcement
                snake_score = 0
                for i in range(4):
                    for j in range(4):
                        if board[i, j] > 0:
                            # Heavier weighting for higher tiles in snake pattern
                            tile_value = np.log2(board[i, j])
                            position_value = 16 - self.snake_pattern[i, j]
                            snake_score += tile_value * position_value
                
                # Normalize and apply with high weight
                shaped_reward += (snake_score / 500) * 10.0
                
                # 3. Massive corner bonus (critical for 2048 strategy)
                if max_tile > 64:
                    if board[3, 3] == max_tile:  # Bottom-right corner (best)
                        shaped_reward += np.log2(max_tile) * 5.0
                    elif board[0, 0] == max_tile:  # Top-left corner (second best)
                        shaped_reward += np.log2(max_tile) * 2.0
                
                # 4. Empty cell bonus (crucial for movement)
                empty_count = np.sum(board == 0)
                shaped_reward += empty_count * 0.5
                
                # 5. Merge potential bonus
                merge_bonus = 0
                # Check horizontal merges
                for i in range(4):
                    for j in range(3):
                        if board[i, j] > 0 and board[i, j] == board[i, j+1]:
                            merge_bonus += board[i, j]
                
                # Check vertical merges
                for j in range(4):
                    for i in range(3):
                        if board[i, j] > 0 and board[i, j] == board[i+1, j]:
                            merge_bonus += board[i, j]
                
                shaped_reward += merge_bonus * 0.01
                
                # 6. Huge bonus for new max tile
                if max_tile > np.max(prev_board):
                    shaped_reward += max_tile * 0.5
                
                # Save the shaped reward
                shaped_rewards.append(shaped_reward)
        
        shaped_rewards = torch.tensor(shaped_rewards, dtype=torch.float32).to(self.device)
        
        # Q-learning update with Double DQN
        # Current Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next actions from current policy
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            # Q-values of next_actions from target network
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            # Target Q-values
            target_q_values = shaped_rewards + (1 - dones) * self.gamma * next_q_values
        
        # Huber loss with prioritization weights
        loss_fn = nn.SmoothL1Loss(reduction='none')
        td_errors = loss_fn(current_q_values, target_q_values)
        weighted_loss = (weights * td_errors).mean()
        
        # Optimization
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities in replay buffer
        with torch.no_grad():
            priorities = (td_errors + 1e-5).detach().cpu().numpy()  # Small constant for stability
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Epsilon decay (faster early on, slower later)
        self.step_counter += 1
        progress = min(self.step_counter / self.decay_steps, 1.0)
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (progress**0.6))
        
        # Target network update
        if self.step_counter % self.target_update_freq == 0:
            self.update_target_model()

# Training and Evaluation Functions
def train_agent(agent, env, num_episodes=10000, max_steps=2000):
    """Train the DQN agent"""
    print(f"Starting training for {num_episodes} episodes...")
    
    # Tracking metrics
    all_rewards = []
    all_max_tiles = []
    epsilon_values = []
    
    best_max_tile = 0
    best_episode_score = 0
    
    # Preload buffer with a few random experiences to avoid empty buffer issues
    print("Initializing experience buffer with random actions...")
    for _ in range(agent.batch_size * 2):
        state = env.reset()
        action = random.randint(0, 3)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
    print(f"Buffer initialized with {len(agent.replay_buffer)} experiences")
    
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()
        state = env.reset()
        total_reward = 0
        step = 0
        done = False
        
        # Episode loop with timeout protection
        episode_timeout = 60  # 60 seconds max per episode
        while not done and step < max_steps:
            # Add timeout check
            if time.time() - episode_start_time > episode_timeout:
                print(f"Episode {episode} timed out after {step} steps")
                break
                
            step += 1
            
            # Get action from agent
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update agent
            agent.train_step()
            
            state = next_state
            total_reward += reward
        
        # Track metrics
        max_tile = np.max(state)
        all_rewards.append(total_reward)
        all_max_tiles.append(max_tile)
        epsilon_values.append(agent.epsilon)
        
        # Update records
        if max_tile > best_max_tile:
            best_max_tile = max_tile
            print(f"NEW BEST TILE: {best_max_tile} in episode {episode}")
        if total_reward > best_episode_score:
            best_episode_score = total_reward
        
        # Print progress for EVERY episode
        episode_time = time.time() - episode_start_time
        print(f"Episode {episode:4d} | Score: {total_reward:7.1f} | Max Tile: {max_tile:5.0f} | " 
              f"Epsilon: {agent.epsilon:.3f} | Time: {episode_time:.1f}s")
        
        # Increase beam search parameters as training progresses and max tile increases
        if max_tile >= 512 and agent.beam_width < 8:
            agent.beam_width = min(8, agent.beam_width + 1)
            agent.search_depth = min(3, agent.search_depth + 1)
            print(f"Increased beam search: width={agent.beam_width}, depth={agent.search_depth}")
        
        # Detailed reporting for milestone episodes
        if episode % 50 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            avg_max_tile = np.mean(all_max_tiles[-50:])
            print(f"SUMMARY - Ep {episode}: Avg Score: {avg_reward:.1f} | Avg Tile: {avg_max_tile:.1f} | Best Tile: {best_max_tile}")
            
            # Evaluation every 500 episodes
            if episode % 500 == 0:
                eval_score, eval_tile = evaluate_agent(agent, env, num_games=5)
                print(f"EVALUATION | Score: {eval_score:.1f} | Max Tile: {eval_tile}")
                
                # Save checkpoint for good performance
                if eval_tile >= 1024:
                    save_model(agent.model, f"results/model_checkpoint_{episode}_{int(eval_tile)}.pt")
                    
        # Save model periodically
        if episode % 1000 == 0:
            save_model(agent.model, f"results/model_episode_{episode}.pt")
    
    print(f"Training completed. Best max tile: {best_max_tile}, Best episode score: {best_episode_score:.1f}")
    return all_rewards, all_max_tiles, epsilon_values

def evaluate_agent(agent, env, num_games=10, render=False):
    """Evaluate the agent's performance without exploration"""
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.01  # Minimal exploration during evaluation
    
    total_scores = []
    total_max_tiles = []
    
    for game in range(num_games):
        state = env.reset()
        done = False
        total_score = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_score += reward
            state = next_state
            
            if render:
                env.render()
                print(f"Action: {env.ACTIONS[action]}, Reward: {reward}")
                time.sleep(0.1)
                
        max_tile = np.max(state)
        total_scores.append(total_score)
        total_max_tiles.append(max_tile)
        
        if render:
            print(f"Game {game+1} - Score: {total_score}, Max Tile: {max_tile}")
    
    # Restore exploration rate
    agent.epsilon = saved_epsilon
    
    return np.mean(total_scores), np.max(total_max_tiles)

def save_model(model, path):
    """Save model weights"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Visualization Functions
def plot_scores(scores, save_path="results/scores.png"):
    """Plot scores over episodes"""
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title("Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    
    # Add rolling average
    if len(scores) > 100:
        rolling_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
        plt.plot(rolling_avg, color='red', label="100-episode average")
        plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def plot_highest_tiles(highest_tiles, save_path="results/highest_tiles.png"):
    """Plot highest tile reached in each episode"""
    plt.figure(figsize=(10, 5))
    plt.plot(highest_tiles)
    plt.title("Max Tile per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Max Tile")
    plt.savefig(save_path)
    plt.close()

def plot_tile_frequency(highest_tiles, save_path="results/tile_frequency.png"):
    """Plot frequency of highest tile values"""
    plt.figure(figsize=(12, 6))
    unique, counts = np.unique(highest_tiles, return_counts=True)
    
    # Convert to log scale for better visualization
    log_values = [int(np.log2(v)) if v > 0 else 0 for v in unique]
    labels = [f"2^{v}={int(2**v)}" for v in log_values]
    
    plt.bar(range(len(counts)), counts, tick_label=labels)
    plt.title("Frequency of Max Tile Values")
    plt.xlabel("Tile Value")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.close()

def plot_epsilon_decay(epsilon_values, save_path="results/epsilon_decay.png"):
    """Plot epsilon decay over episodes"""
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_values)
    plt.title("Epsilon Decay Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.savefig(save_path)
    plt.close()

# Main function
def main():
    env = Game2048Env()
    
    # Create results folder if it doesn't exist
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    
    # Initialize model, target model and replay buffer
    model = HybridDQN()
    target_model = HybridDQN()
    replay_buffer = PrioritizedReplayBuffer(capacity=200000)
    
    # Start with a smaller batch size for stability
    initial_batch_size = 128  # Start smaller for first 500 episodes
    agent = DQNAgent(
        env=env,
        model=model,
        target_model=target_model,
        replay_buffer=replay_buffer,
        batch_size=initial_batch_size,  # Smaller initial batch size
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.001,
        decay_steps=150000,
        target_update_freq=250,
        learning_rate=1e-3
    )
    
    # Train the agent
    print("Starting new training run with aggressive hybrid CNN-Transformer model.")
    print(f"Using initial batch size: {initial_batch_size}")
    scores, highest_tiles, epsilon_values = train_agent(agent, env, num_episodes=10000, max_steps=2000)
    
    # Save final model
    final_model_path = os.path.join(results_folder, "hybrid_cnn_transformer_final.pt")
    save_model(agent.model, final_model_path)
    
    # Plot results
    plot_scores(scores)
    plot_highest_tiles(highest_tiles)
    plot_tile_frequency(highest_tiles)
    plot_epsilon_decay(epsilon_values)
    
    # Final evaluation
    print("\nFinal evaluation:")
    eval_score, eval_tile = evaluate_agent(agent, env, num_games=20)
    print(f"Average Score: {eval_score:.1f}, Best Tile: {eval_tile}")
    
    print(f"\nTraining completed. Results saved to {results_folder}")
    print(f"Highest Score: {max(scores):.1f}")
    print(f"Highest Tile: {max(highest_tiles)}")

if __name__ == "__main__":
    import time
    main()
