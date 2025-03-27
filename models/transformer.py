import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, board_size=4, num_actions=4, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        
        self.board_size = board_size
        self.embedding = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model * board_size * board_size, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Actor (policy) and critic (value) heads
        self.actor = nn.Linear(64, num_actions)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape and embed the board
        x = x.view(batch_size, self.board_size * self.board_size, 1)
        x = self.embedding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Flatten and process through FC layers
        x = x.reshape(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Get action probabilities and state value
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value