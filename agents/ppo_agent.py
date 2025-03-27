import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else
                     "cpu")

class PPOMemory:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, action_prob, reward, next_state, done):
        self.buffer.append((state, action, action_prob, reward, next_state, done))
    
    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, probs, rewards, next_states, dones = [], [], [], [], [], []
        
        for i in indices:
            s, a, p, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            probs.append(p)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            
        # Ensure we have numpy arrays (not tensors)
        if isinstance(states[0], torch.Tensor):
            states = [s.cpu().numpy() if s.device.type == 'mps' else s.numpy() for s in states]
        if isinstance(actions[0], torch.Tensor):
            actions = [a.cpu().numpy() if a.device.type == 'mps' else a.numpy() for a in actions]
        if isinstance(probs[0], torch.Tensor):
            probs = [p.cpu().numpy() if p.device.type == 'mps' else p.numpy() for p in probs]
        if isinstance(rewards[0], torch.Tensor):
            rewards = [r.cpu().numpy() if r.device.type == 'mps' else r.numpy() for r in rewards]
        if isinstance(next_states[0], torch.Tensor):
            next_states = [s.cpu().numpy() if s.device.type == 'mps' else s.numpy() for s in next_states]
        if isinstance(dones[0], torch.Tensor):
            dones = [d.cpu().numpy() if d.device.type == 'mps' else d.numpy() for d in dones]
            
        return np.array(states), np.array(actions), np.array(probs), \
               np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

class ActorNetwork(nn.Module):
    def __init__(self, input_dim=16, output_dim=4):
        super(ActorNetwork, self).__init__()
        # Deeper network with more capacity
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Add some dropout for regularization
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # Handle batched and non-batched inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False
            
        x = self.relu(self.fc1(x))
        if x.shape[0] > 1:  # Apply batch norm only if we have a batch
            x = self.bn1(x)
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.dropout(x)
        
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        
        if single_input:
            return x.squeeze(0)
        return x

class CriticNetwork(nn.Module):
    def __init__(self, input_dim=16):
        super(CriticNetwork, self).__init__()
        # Similarly enhanced architecture
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Handle batched and non-batched inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False
            
        x = self.relu(self.fc1(x))
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.dropout(x)
        
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        if single_input:
            return x.squeeze(0)
        return x

class PPOAgent:
    def __init__(self, state_dim=16, action_dim=4):
        """
        PPO Agent for 2048 game - Aggressive exploration version
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        print(f"Aggressive PPOAgent using device: {self.device}")
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # More aggressive learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=8e-4)  # Increased from 5e-4
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2e-3)  # Increased from 1e-3
        
        # More aggressive hyperparameters
        self.gamma = 0.995  # Higher discount factor for long-term rewards
        self.clip_epsilon = 0.3  # More aggressive clipping (was 0.2)
        self.epochs = 8  # More epochs (was 5)
        self.batch_size = 256  # Larger batch size (was 128)
        self.value_coef = 0.4
        self.entropy_coef = 0.05  # Much higher entropy coefficient for exploration
        
        # Much more aggressive exploration strategy
        self.exploration_rate = 0.7  # Higher initial exploration (was 0.4)
        self.exploration_decay = 0.995  # Slower decay (was 0.99)
        self.min_exploration_rate = 0.15  # Higher minimum exploration (was 0.05)
        
        # Track highest tile achieved for reward shaping
        self.highest_tile_seen = 2  # Start at 2 (the lowest tile)
        self.highest_tile_history = []  # Track progress over time
        
        # Track board states to reward novel positions
        self.seen_states = set()
        self.novelty_factor = 0.2  # Reward for novel states
        
        # Experience replay with larger capacity
        self.memory = PPOMemory(50000)  # Larger memory (was 20000)
        
        # Heuristic weights
        self.heuristic_weight = 0.3  # Weight for heuristic evaluation
        
    def normalize_state(self, state):
        """Normalize state values for better learning with enhanced features."""
        # Convert to float and apply log2
        log_state = np.zeros_like(state, dtype=np.float32)
        mask = state > 0
        log_state[mask] = np.log2(state[mask])
        
        # Normalize to 0-1 range
        if np.max(log_state) > 0:
            log_state = log_state / 15.0  # Max possible is 2^15=32768 (aim higher than 2048)
        
        return log_state
    
    def get_action(self, state, valid_moves=None):
        """Select an action based on current policy."""
        try:
            # Normalize state
            norm_state = self.normalize_state(state)
            state_tensor = torch.FloatTensor(norm_state).to(self.device)
            
            # Get action probabilities from actor
            self.actor.eval()
            with torch.no_grad():
                action_probs = self.actor(state_tensor)
            self.actor.train()
            
            # Apply mask for valid moves if provided
            if valid_moves is not None:
                # Create mask with negative infinity for invalid moves
                mask = torch.tensor([float('-inf') if not valid else 0.0 
                                    for valid in valid_moves], dtype=torch.float32).to(self.device)
                # Apply mask to action probabilities (logits)
                masked_probs = torch.log(action_probs + 1e-10) + mask
                # Get action using masked probabilities
                dist = torch.distributions.Categorical(logits=masked_probs)
                action = dist.sample()
                action_prob = dist.log_prob(action)
                return action.item(), action_prob.item()
            
            # If no valid moves specified, sample from the distribution
            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
            action_prob = dist.log_prob(action)
            
            return action.item(), action_prob.item()
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Fallback to random action
            return random.randint(0, self.action_dim - 1), 0.0
    
    def remember(self, state, action, action_prob, reward, next_state, done):
        """Store experience with aggressive reward shaping."""
        # Get max tile values
        current_max_tile = np.max(state)
        next_max_tile = np.max(next_state)
        
        # Track highest tile seen
        if next_max_tile > self.highest_tile_seen:
            tile_bonus = 5.0 * (np.log2(next_max_tile) - np.log2(self.highest_tile_seen))
            self.highest_tile_seen = next_max_tile
            self.highest_tile_history.append(next_max_tile)
            reward += tile_bonus  # Huge bonus for new highest tile
            print(f"New highest tile: {next_max_tile}, bonus: {tile_bonus:.2f}")
        
        # Penalize regression in tile values
        if next_max_tile < current_max_tile:
            regression_penalty = -2.0 * (np.log2(current_max_tile) - np.log2(next_max_tile))
            reward += regression_penalty
        
        # Bonus for higher tiles on the board (encourages building big tiles)
        top_tiles = np.sort(next_state.flatten())[-4:]  # Get top 4 tiles
        high_tile_bonus = 0.1 * sum(np.log2(t) for t in top_tiles if t > 0)
        reward += high_tile_bonus
        
        # Reward for novel board states (exploration)
        state_hash = hash(next_state.tobytes())
        if state_hash not in self.seen_states:
            self.seen_states.add(state_hash)
            reward += self.novelty_factor
        
        # Apply heuristic bonuses
        heuristic_score = self.evaluate_heuristic(next_state)
        reward += self.heuristic_weight * heuristic_score
        
        # Store enhanced experience
        self.memory.add(state, action, action_prob, reward, next_state, done)
        
    def evaluate_heuristic(self, state):
        """Evaluate board state using heuristics."""
        board = state.reshape(4, 4)
        score = 0
        
        # 1. Reward for monotonicity (tiles increasing toward a corner)
        # Check for monotonicity in all directions and take the best
        monotonicity_score = max(
            self.monotonicity(board, 1, 1),   # Increasing to bottom-right
            self.monotonicity(board, 1, -1),  # Increasing to bottom-left
            self.monotonicity(board, -1, 1),  # Increasing to top-right
            self.monotonicity(board, -1, -1)  # Increasing to top-left
        )
        score += 2.0 * monotonicity_score
        
        # 2. Reward for keeping high values in corners
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        max_corner = max(corners)
        if max_corner == np.max(board):
            score += 1.0
        
        # 3. Penalize for scattered high values
        high_tiles = board[board >= 8]  # Tiles >= 8
        if len(high_tiles) > 0:
            smoothness_penalty = -0.1 * len(high_tiles)
            score += smoothness_penalty
        
        return score
    
    def monotonicity(self, board, row_dir, col_dir):
        """Calculate how monotonic the board is in a given direction."""
        score = 0
        
        # Starting positions based on directions
        if row_dir > 0:
            rows = range(4)
        else:
            rows = range(3, -1, -1)
            
        if col_dir > 0:
            cols = range(4)
        else:
            cols = range(3, -1, -1)
        
        # Check rows
        for r in rows:
            for c in range(3):
                if board[r, c] > 0 and board[r, c+1] > 0:
                    if row_dir > 0:
                        score += int(board[r, c] <= board[r, c+1])
                    else:
                        score += int(board[r, c] >= board[r, c+1])
        
        # Check columns
        for c in cols:
            for r in range(3):
                if board[r, c] > 0 and board[r+1, c] > 0:
                    if col_dir > 0:
                        score += int(board[r, c] <= board[r+1, c])
                    else:
                        score += int(board[r, c] >= board[r+1, c])
        
        return score / 24.0  # Normalize
    
    def update(self):
        """Update policy and value networks."""
        if len(self.memory) < self.batch_size:
            return
        
        try:
            # Sample from memory
            states, actions, old_probs, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # Normalize states
            norm_states = np.array([self.normalize_state(s) for s in states])
            norm_next_states = np.array([self.normalize_state(s) for s in next_states])
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(norm_states).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            old_probs_tensor = torch.FloatTensor(old_probs).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            next_states_tensor = torch.FloatTensor(norm_next_states).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)
            
            # Calculate returns and advantages
            with torch.no_grad():
                values = self.critic(states_tensor).squeeze()
                next_values = self.critic(next_states_tensor).squeeze()
                
                # Handle scalar case
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                if next_values.dim() == 0:
                    next_values = next_values.unsqueeze(0)
                
                # Calculate returns using GAE
                returns = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)
                advantages = returns - values
                
                # Normalize advantages
                if advantages.shape[0] > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update loop
            for _ in range(self.epochs):
                # Get current policy distributions and values
                new_probs = self.actor(states_tensor)
                dist = torch.distributions.Categorical(probs=new_probs)
                new_log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()
                
                value_pred = self.critic(states_tensor).squeeze()
                
                # Handle scalar case
                if value_pred.dim() == 0:
                    value_pred = value_pred.unsqueeze(0)
                if returns.dim() == 0:
                    returns = returns.unsqueeze(0)
                
                # Calculate ratio and clipped ratio for PPO
                ratio = torch.exp(new_log_probs - old_probs_tensor)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                # Calculate actor and critic losses
                actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                value_loss = torch.nn.functional.mse_loss(value_pred, returns)
                
                # Combined loss
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Skip if NaN
                if torch.isnan(loss):
                    continue
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
            
            # Clear memory after update
            self.memory.clear()
            
        except Exception as e:
            print(f"Error in update: {e}")
    
    def save(self, path):
        """Save model weights."""
        directory = os.path.dirname(path)
        if (directory and not os.path.exists(directory)):
            os.makedirs(directory)
            
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model weights."""
        if not os.path.exists(path):
            print(f"Model file {path} not found")
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {path}")
        return True
