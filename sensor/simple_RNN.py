import torch 
import torch.nn as nn   
import torch.functional as F


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, dropout=0.2, output_size=1):
        super(SimpleRNN, self).__init__()
        
        # Single LSTM stack
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*seq_len, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM layer
        x, _ = self.lstm(x)
        
        # Fully connected layers
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x