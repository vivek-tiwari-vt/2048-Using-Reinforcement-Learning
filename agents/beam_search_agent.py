import numpy as np
import copy
import random
from environment.game_2048 import Game2048Env
import os

class BeamSearchAgent:
    """
    Optimized Beam Search Agent for 2048 game.
    Balances search efficiency with effective board evaluation.
    """
    
    def __init__(self, beam_width=10, search_depth=15):
        """
        Initialize the beam search agent with optimized parameters
        
        Args:
            beam_width: Number of candidate states to maintain (k)
            search_depth: How deep to search in the game tree (d)
        """
        self.beam_width = beam_width  # Reduced from 15 to 10
        self.search_depth = search_depth  # Reduced from 30 to 15
        self.action_names = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}
        
        # Game phase detection
        self.early_game_threshold = 512
        self.mid_game_threshold = 1024
        
        # Precompute snake patterns and gradients for faster lookup
        self._init_patterns()
    
    def _init_patterns(self):
        """Precompute evaluation patterns for faster lookup"""
        # Snake patterns
        self.snake_patterns = [
            # Standard snake (decreasing values in zig-zag)
            np.array([
                [15, 14, 13, 12],
                [8,  9,  10, 11],
                [7,  6,  5,  4],
                [0,  1,  2,  3]
            ]),
            # Corner-based snake
            np.array([
                [15, 14, 13, 12],
                [11, 10, 9,  8],
                [7,  6,  5,  4],
                [3,  2,  1,  0]
            ])
        ]
        
        # Gradient patterns
        self.gradients = [
            np.array([  # Top-left to bottom-right
                [4, 3, 2, 1],
                [5, 4, 3, 2],
                [6, 5, 4, 3],
                [7, 6, 5, 4]
            ]),
            np.array([  # Bottom-left to top-right
                [7, 6, 5, 4],
                [6, 5, 4, 3],
                [5, 4, 3, 2],
                [4, 3, 2, 1]
            ])
        ]
        
        # Corner positions for fast lookup
        self.corners = [(0,0), (0,3), (3,0), (3,3)]
    
    def get_action(self, state, valid_moves=None):
        """Find the best action using optimized beam search"""
        # Reshape state if it's flattened
        if len(state.shape) == 1:
            board_size = int(np.sqrt(state.shape[0]))
            state = state.reshape(board_size, board_size)
        
        # Use lightweight board representation instead of full environment
        board = state.copy()
        
        # Get valid moves
        if valid_moves is None:
            # Simple valid move check without creating full environment
            valid_moves = self._check_valid_moves(board)
        
        if not any(valid_moves):
            # No valid moves, return a default action and probability
            return 0, 0.5
        
        # Fast path: if there's only one valid move, take it
        if sum(valid_moves) == 1:
            action = valid_moves.index(True)
            return action, 1.0
            
        # Get the game phase for dynamic evaluation
        max_tile = np.max(board)
        game_phase = self._determine_game_phase(max_tile)
        
        # Adjust search parameters based on board complexity
        empty_count = np.sum(board == 0)
        if empty_count <= 4:  # Critical situation, search deeper
            actual_depth = min(self.search_depth + 5, 25)
        elif empty_count >= 10:  # Early game, can use shallower search
            actual_depth = min(self.search_depth - 5, 10)
        else:
            actual_depth = self.search_depth
        
        # Initialize with root node - directly evaluate actions
        initial_candidates = []
        
        # Try each valid action
        for action in range(4):
            if valid_moves[action]:
                # Make a move without creating a full environment
                new_board, move_score, valid = self._make_move(board.copy(), action)
                if valid:
                    # Add a random tile - just do this once per action for speed
                    self._add_random_tile(new_board)
                    initial_candidates.append({
                        'board': new_board,
                        'path': [action],
                        'score': self._fast_evaluate(new_board, game_phase)
                    })
        
        # If no candidates, return random action
        if not initial_candidates:
            valid_actions = [i for i in range(4) if valid_moves[i]]
            return random.choice(valid_actions), 0.5
        
        # Sort by score and keep top k candidates
        candidates = sorted(initial_candidates, key=lambda x: x['score'], reverse=True)
        candidates = candidates[:self.beam_width]
        
        # Expand the beam for specified depth using fast evaluation
        for depth in range(1, actual_depth):
            next_candidates = []
            
            # Use lighter evaluation for deeper levels
            use_fast_eval = depth > 3
            
            # Process each candidate
            for candidate in candidates:
                board = candidate['board']
                path = candidate['path']
                
                # Check valid moves for this board
                candidate_valid_moves = self._check_valid_moves(board)
                
                # Try each valid action
                for action in range(4):
                    if candidate_valid_moves[action]:
                        new_board, move_score, valid = self._make_move(board.copy(), action)
                        if valid:
                            # Add random tile - reduced randomness for speed
                            self._add_random_tile(new_board)
                            
                            # Use appropriate evaluation function
                            if use_fast_eval:
                                eval_score = self._fast_evaluate(new_board, game_phase)
                            else:
                                eval_score = self._evaluate_state(new_board, game_phase)
                            
                            next_candidates.append({
                                'board': new_board,
                                'path': path + [action],
                                'score': eval_score
                            })
            
            # If no next candidates, break
            if not next_candidates:
                break
                
            # Keep top k candidates
            candidates = sorted(next_candidates, key=lambda x: x['score'], reverse=True)
            candidates = candidates[:self.beam_width]
        
        # Return the first action from the best path
        best_action = candidates[0]['path'][0] if candidates else 0
        
        # Return the action and a dummy probability
        return best_action, 1.0
    
    def _check_valid_moves(self, board):
        """Fast check for valid moves without creating an environment"""
        valid_moves = [False, False, False, False]
        
        # Check each direction
        for action in range(4):
            new_board, _, valid = self._make_move(board.copy(), action)
            valid_moves[action] = valid
        
        return valid_moves
    
    def _make_move(self, board, action):
        """
        Make a move on the board without using the environment
        Returns: (new_board, score_gained, move_was_valid)
        """
        original_board = board.copy()
        score = 0
        
        # Pre-process board based on action
        if action == 0:  # LEFT
            pass  # No preprocessing needed
        elif action == 1:  # UP
            board = board.T
        elif action == 2:  # RIGHT
            board = np.fliplr(board)
        elif action == 3:  # DOWN
            board = np.fliplr(board.T)
        
        # Process each row (move left)
        for i in range(board.shape[0]):
            # Get non-zero values
            row = board[i]
            non_zero = row[row != 0]
            
            if len(non_zero) == 0:
                continue
                
            # Merge tiles
            merged_row = []
            skip_next = False
            merge_score = 0
            
            for j in range(len(non_zero)):
                if skip_next:
                    skip_next = False
                    continue
                
                if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                    merged_value = non_zero[j] * 2
                    merged_row.append(merged_value)
                    merge_score += merged_value
                    skip_next = True
                else:
                    merged_row.append(non_zero[j])
            
            # Pad with zeros
            merged_row = np.array(merged_row + [0] * (board.shape[1] - len(merged_row)))
            board[i] = merged_row
            score += merge_score
        
        # Post-process board based on action
        if action == 0:  # LEFT
            pass  # No postprocessing needed
        elif action == 1:  # UP
            board = board.T
        elif action == 2:  # RIGHT
            board = np.fliplr(board)
        elif action == 3:  # DOWN
            board = board.T
            board = np.fliplr(board)
        
        # Check if the move was valid
        move_valid = not np.array_equal(original_board, board)
        
        return board, score, move_valid
    
    def _add_random_tile(self, board):
        """Add a random tile to the board (2 or 4)"""
        empty_cells = np.where(board == 0)
        if len(empty_cells[0]) > 0:
            # Pick a random empty cell
            idx = random.randint(0, len(empty_cells[0]) - 1)
            row, col = empty_cells[0][idx], empty_cells[1][idx]
            
            # 90% chance for a 2, 10% chance for a 4
            board[row, col] = 2 if random.random() < 0.9 else 4
    
    def _determine_game_phase(self, max_tile):
        """Determine the phase of the game based on highest tile"""
        if max_tile < self.early_game_threshold:
            return "early"
        elif max_tile < self.mid_game_threshold:
            return "mid"
        else:
            return "late"
    
    def _fast_evaluate(self, board, game_phase):
        """
        Very fast evaluation function for deeper search levels
        Only uses the most important heuristics
        """
        # Count empty tiles
        empty_count = np.sum(board == 0)
        empty_score = empty_count * 10.0
        
        # Highest tile bonus
        max_tile = np.max(board)
        max_score = np.log2(max_tile) * 2.0 if max_tile > 0 else 0
        
        # Simple check for monotonicity using corners
        corner_scores = []
        for corner in self.corners:
            score = board[corner[0], corner[1]] * 2
            if score > 0:  # If there's a value in the corner
                corner_scores.append(score)
        
        corner_score = max(corner_scores) if corner_scores else 0
        
        # Merge potential (quick check)
        merge_score = 0
        for i in range(board.shape[0]):
            for j in range(board.shape[1] - 1):
                if board[i, j] == board[i, j+1] and board[i, j] > 0:
                    merge_score += 1
        
        for i in range(board.shape[0] - 1):
            for j in range(board.shape[1]):
                if board[i, j] == board[i+1, j] and board[i, j] > 0:
                    merge_score += 1
        
        return empty_score + max_score + corner_score + merge_score * 2
    
    def _evaluate_state(self, board, game_phase):
        """
        Optimized evaluation function for board state
        """
        # Simplified weights based on game phase
        if game_phase == "early":
            empty_weight = 15.0
            max_tile_weight = 1.0
            corner_weight = 2.0
            merge_weight = 2.0
        elif game_phase == "mid":
            empty_weight = 10.0
            max_tile_weight = 1.5
            corner_weight = 2.5
            merge_weight = 1.5
        else:  # late
            empty_weight = 8.0
            max_tile_weight = 2.0
            corner_weight = 3.0
            merge_weight = 1.0
        
        # 1. Count empty tiles
        empty_count = np.sum(board == 0)
        empty_score = empty_count * empty_weight
        
        # Add penalty for critical low empty count
        if empty_count <= 2:
            empty_score -= 10.0
            
        # 2. Value of highest tile
        max_tile = np.max(board)
        max_score = np.log2(max_tile) * max_tile_weight if max_tile > 0 else 0
        
        # Progressive bonus for milestone tiles
        if max_tile >= 512:
            max_score *= 1.2
        if max_tile >= 1024:
            max_score *= 1.5
        if max_tile >= 2048:
            max_score *= 2.0
        
        # 3. Corner placement bonus (high tiles in corner)
        corner_bonus = self._calculate_corner_bonus(board) * corner_weight
        
        # 4. Merge potential - reward states with many possible merges
        merge_potential = self._calculate_merge_potential(board) * merge_weight
        
        # 5. Use only one snake pattern for speed
        snake_score = 0
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] > 0:
                    value = np.log2(board[i, j])
                    snake_score += value * self.snake_patterns[0][i, j]
        snake_score /= 100.0
        
        # Combine scores
        return empty_score + max_score + corner_bonus + merge_potential + snake_score

    def _calculate_corner_bonus(self, board):
        """Optimized corner bonus calculation"""
        # Just check the corners directly
        corner_values = [board[0,0], board[0,3], board[3,0], board[3,3]]
        max_corner = max(corner_values)
        
        if max_corner <= 0:
            return 0
            
        # Basic bonus based on corner value
        return np.log2(max_corner) * 2.0

    def _calculate_merge_potential(self, board):
        """Optimized merge potential calculation"""
        merge_potential = 0
        
        # Check horizontally
        for i in range(board.shape[0]):
            for j in range(board.shape[1]-1):
                if board[i,j] > 0 and board[i,j] == board[i,j+1]:
                    merge_potential += np.log2(board[i,j])
        
        # Check vertically
        for i in range(board.shape[0]-1):
            for j in range(board.shape[1]):
                if board[i,j] > 0 and board[i,j] == board[i+1,j]:
                    merge_potential += np.log2(board[i,j])
        
        return merge_potential

    def remember(self, *args):
        """Dummy method to match the agent interface"""
        pass
    
    def update(self):
        """Dummy method to match the agent interface"""
        pass
    
    def save(self, path):
        """
        Save the beam search agent configuration
        
        Args:
            path: Path to save the configuration
        """
        config = {
            'beam_width': self.beam_width,
            'search_depth': self.search_depth,
            'early_game_threshold': self.early_game_threshold,
            'mid_game_threshold': self.mid_game_threshold
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the configuration
        import json
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Beam Search configuration saved to {path}")
        
        # Also save as a README for human readability
        readme_path = os.path.join(os.path.dirname(path), 
                                  f"beam_search_config_readme_{self.beam_width}_{self.search_depth}.txt")
        with open(readme_path, 'w') as f:
            f.write(f"Beam Search Agent Configuration\n")
            f.write(f"==============================\n\n")
            f.write(f"Beam Width: {self.beam_width}\n")
            f.write(f"Search Depth: {self.search_depth}\n")
            f.write(f"Early Game Threshold: {self.early_game_threshold}\n")
            f.write(f"Mid Game Threshold: {self.mid_game_threshold}\n")
            f.write(f"\nSaved at: {path}\n")
            f.write(f"\nThis configuration achieved good results in training.\n")
            f.write(f"To recreate this agent, use:\n")
            f.write(f"agent = BeamSearchAgent(beam_width={self.beam_width}, search_depth={self.search_depth})")

    @classmethod
    def load(cls, path):
        """
        Load a beam search agent configuration
        
        Args:
            path: Path to the saved configuration
            
        Returns:
            BeamSearchAgent: An agent with the loaded configuration
        """
        import json
        with open(path, 'r') as f:
            config = json.load(f)
        
        agent = cls(
            beam_width=config.get('beam_width', 10), 
            search_depth=config.get('search_depth', 15)
        )
        
        # Set other parameters if present
        if 'early_game_threshold' in config:
            agent.early_game_threshold = config['early_game_threshold']
        if 'mid_game_threshold' in config:
            agent.mid_game_threshold = config['mid_game_threshold']
        
        print(f"Beam Search configuration loaded from {path}")
        return agent