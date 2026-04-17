import torch
import torch.nn as nn
import numpy as np

class LSTMAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LSTMAgent, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, action_dim)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, state_dim)
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        action = torch.tanh(self.fc(out[:, -1, :]))
        return action, hidden
