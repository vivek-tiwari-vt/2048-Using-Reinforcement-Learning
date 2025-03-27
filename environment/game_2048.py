import numpy as np
import random

class Game2048Env:
    """
    Environment class for the 2048 game designed for reinforcement learning.
    Follows the exact rules of the original 2048 game.
    """
    
    # Action mapping
    ACTIONS = {
        0: "LEFT",
        1: "UP",
        2: "RIGHT",
        3: "DOWN"
    }
    
    def __init__(self, size=4):
        """
        Initialize the 2048 game environment.
        
        Args:
            size (int): Size of the game board (size × size)
        """
        self.size = size
        self.highest_tile = 0
        self.reset()
    
    def reset(self):
        """
        Reset the game board and add two initial tiles.
        
        Returns:
            numpy.ndarray: Initial state of the game
        """
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self.highest_tile = 0
        
        # Add two initial tiles
        self.add_new_tile()
        self.add_new_tile()
        
        # Update highest tile
        self.highest_tile = np.max(self.board)
        
        return self.get_state()
    
    def get_state(self):
        """
        Return the current state of the game.
        
        Returns:
            numpy.ndarray: Current game board state
        """
        return self.board.flatten()  # Return flattened board for RL
    
    def add_new_tile(self):
        """Add a new tile (2 or 4) to a random empty position."""
        if np.any(self.board == 0):
            # Find empty positions
            empty_positions = list(zip(*np.where(self.board == 0)))
            pos = random.choice(empty_positions)
            
            # 90% chance for a 2, 10% chance for a 4
            self.board[pos] = 2 if random.random() < 0.9 else 4
    
    def get_valid_moves(self):
        """
        Determine which moves are valid in the current state.
        
        Returns:
            list: Boolean list where True indicates a valid move [LEFT, UP, RIGHT, DOWN]
        """
        valid_moves = []
        
        # Save the current board state
        original_board = self.board.copy()
        original_score = self.score
        
        # Check each possible move
        for action in range(4):
            # Try the move
            self._execute_move(action)
            
            # If the board changed, the move is valid
            valid = not np.array_equal(original_board, self.board)
            valid_moves.append(valid)
            
            # Restore the original board state
            self.board = original_board.copy()
            self.score = original_score
            
        return valid_moves
    
    def _execute_move(self, action):
        """Execute the specified move without adding a new tile."""
        if action == 0:  # Left
            self._move_left()
        elif action == 1:  # Up
            self.board = self.board.T
            self._move_left()
            self.board = self.board.T
        elif action == 2:  # Right
            self.board = np.fliplr(self.board)
            self._move_left()
            self.board = np.fliplr(self.board)
        elif action == 3:  # Down
            self.board = self.board.T
            self.board = np.fliplr(self.board)
            self._move_left()
            self.board = np.fliplr(self.board)
            self.board = self.board.T
    
    def _move_left(self):
        """
        Perform a left move and merge tiles.
        Updates the board in-place following 2048 rules:
        1. Each tile moves as far left as possible
        2. If two tiles of the same value collide, they merge
        3. Each tile can only merge once per move
        """
        changed = False
        
        for i in range(self.size):
            # Get the row and make a copy
            original_row = self.board[i].copy()
            
            # First, remove all zeros and compact the row
            row = self.board[i][self.board[i] != 0]
            
            # If the row is empty, continue
            if len(row) == 0:
                continue
                
            # Create a new row to handle merging
            new_row = []
            skip_next = False
            
            # Process each tile for merging
            for j in range(len(row)):
                if skip_next:
                    skip_next = False
                    continue
                
                # Check if we can merge with the next tile
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    merged_value = row[j] * 2
                    new_row.append(merged_value)
                    self.score += merged_value  # Add score
                    skip_next = True
                    changed = True
                else:
                    new_row.append(row[j])
            
            # Convert to numpy array and pad with zeros
            new_row = np.array(new_row, dtype=np.int32)
            new_row = np.pad(new_row, (0, self.size - len(new_row)), 'constant')
            
            # Update the board
            self.board[i] = new_row
            
            # Check if the row changed
            if not np.array_equal(original_row, new_row):
                changed = True
        
        return changed
    
    def step(self, action):
        """
        Take a step in the environment by executing the specified action.
        
        Args:
            action (int): The action to take (0=LEFT, 1=UP, 2=RIGHT, 3=DOWN)
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Store previous state for comparison and reward calculation
        prev_score = self.score
        prev_board = self.board.copy()
        
        # Execute the move
        self._execute_move(action)
        
        # Determine if the move changed the board
        valid_move = not np.array_equal(prev_board, self.board)
        
        # Add new tile if the move was valid
        if valid_move:
            self.add_new_tile()
        
        # Calculate reward
        reward = self._calculate_reward(valid_move, prev_board, prev_score)
        
        # Check if game is over
        self.game_over = self.is_game_over()
        
        # Update highest tile
        current_highest = np.max(self.board)
        if current_highest > self.highest_tile:
            self.highest_tile = current_highest
        
        # Return state, reward, done, info
        return self.get_state(), reward, self.game_over, {
            "score": self.score, 
            "valid_move": valid_move,
            "highest_tile": self.highest_tile
        }
    
    def _calculate_reward(self, valid_move, prev_board, prev_score):
        """
        Calculate the reward based on the game state changes.
        
        Args:
            valid_move (bool): Whether the move was valid
            prev_board (numpy.ndarray): Board state before the move
            prev_score (int): Score before the move
            
        Returns:
            float: Calculated reward
        """
        # Base reward: score increase (more weight on higher scores)
        score_diff = self.score - prev_score
        reward = score_diff / 4.0  # Increase the weight of score gains
        
        # Major bonus for reaching a new highest tile
        if self.highest_tile > np.max(prev_board):
            # Exponential reward for higher tiles
            reward += 2.0 * np.log2(self.highest_tile)
            
            # Extra bonus for milestone tiles
            if self.highest_tile >= 256:
                reward += 50
            if self.highest_tile >= 512:
                reward += 100
            if self.highest_tile >= 1024:
                reward += 200
            if self.highest_tile >= 2048:
                reward += 500
        
        # Penalty for invalid moves
        if not valid_move:
            reward -= 2.0  # Increase penalty
        
        # Strategic board evaluation
        # Bonus for keeping empty spaces (crucial for high tile strategies)
        empty_before = np.sum(prev_board == 0)
        empty_after = np.sum(self.board == 0)
        reward += (empty_after - empty_before) * 0.5  # Increased weight
        
        # Encourage "edge building" strategy (keeping high values on edges)
        board_2d = self.board.reshape(self.size, self.size)
        edge_sum = np.sum(board_2d[0,:]) + np.sum(board_2d[-1,:]) + \
                   np.sum(board_2d[:,0]) + np.sum(board_2d[:,-1])
        
        # Reward for keeping high values on edges
        reward += (edge_sum / np.sum(self.board)) * 1.0
        
        # Severe penalty for being close to losing
        if empty_after <= 2:  # Critical situation
            reward -= 2.0
        
        # Encourage monotonicity (having tiles in ascending/descending order)
        # This helps build organized patterns
        for i in range(self.size):
            row = board_2d[i,:]
            col = board_2d[:,i]
            
            # Reward for having rows and columns in ascending/descending order
            row_ordered = sum(row[j] >= row[j-1] for j in range(1, self.size) if row[j] > 0 and row[j-1] > 0)
            col_ordered = sum(col[j] >= col[j-1] for j in range(1, self.size) if col[j] > 0 and col[j-1] > 0)
            
            reward += (row_ordered + col_ordered) * 0.1
            
        return reward
    
    def is_game_over(self):
        """
        Check if the game is over (no more valid moves).
        
        Returns:
            bool: True if game is over, False otherwise
        """
        # If there are valid moves, game is not over
        valid_moves = self.get_valid_moves()
        return not any(valid_moves)
    
    def render(self, mode='human'):
        """
        Render the current state of the game.
        
        Args:
            mode (str): Rendering mode ('human' for terminal output)
        """
        if mode == 'human':
            print('-' * (5 * self.size + 1))
            for row in self.board:
                print('|', end='')
                for tile in row:
                    val = int(tile)
                    if val == 0:
                        print('    |', end='')
                    else:
                        print(f'{val:4d}|', end='')
                print()
                print('-' * (5 * self.size + 1))
            print(f"Score: {self.score}")
            print(f"Highest Tile: {self.highest_tile}")
            print()
    
    def _evaluate_pattern(self):
        """Evaluate board pattern for strategic positioning"""
        board = self.board.reshape(self.size, self.size)
        pattern_score = 0
        
        # Snake pattern (zigzag) is good for 2048
        snake_multipliers = np.array([
            [16, 15, 14, 13],
            [9, 10, 11, 12],
            [8, 7, 6, 5],
            [1, 2, 3, 4]
        ])
        
        # Corner pattern (monotonically decreasing from corner)
        corner_multipliers = np.array([
            [16, 8, 4, 2],
            [8, 4, 2, 1],
            [4, 2, 1, 0.5],
            [2, 1, 0.5, 0.25]
        ])
        
        # Calculate pattern scores
        snake_score = np.sum(board * snake_multipliers) / 100.0
        corner_score = np.sum(board * corner_multipliers) / 100.0
        
        # Return highest pattern score
        return max(snake_score, corner_score)
    
    def simulate_move(self, state, action):
        """
        Simulate a move without modifying the actual game state.
        
        Args:
            state (numpy.ndarray): The state to simulate from (flattened)
            action (int): The action to take
            
        Returns:
            list: List of tuples (next_state, reward, done) for all possible next states
        """
        # Save current board and score
        temp_board = self.board.copy()
        temp_score = self.score
        
        # Setup for simulation
        state_2d = state.reshape(self.size, self.size)
        self.board = state_2d.copy()
        original_score = self.score
        
        # Execute move
        self._execute_move(action)
        changed = not np.array_equal(state_2d, self.board)
        
        possible_states = []
        if changed:
            empty = list(zip(*np.where(self.board == 0)))
            for pos in empty:
                for tile in [2, 4]:
                    # Create new state with the tile added
                    new_state = self.board.copy()
                    new_state[pos] = tile
                    
                    # Calculate reward
                    reward = self._calculate_reward(True, state_2d, original_score)
                    
                    # Check if this new state is terminal
                    self.board = new_state.copy()
                    done = self.is_game_over()
                    
                    # Add to possible states (flatten state for consistency)
                    possible_states.append((new_state.flatten(), reward, done))
        
        # Restore original state
        self.board = temp_board
        self.score = temp_score
        return possible_states