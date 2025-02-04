import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_size, num_classes, seq_len, conv_filters=64, lstm_units=128, dropout_rate=0.3):
        super(TimeSeriesClassifier, self).__init__()
        
        # CNN Feature Extraction
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_filters, conv_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_filters*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM Temporal Processing
        self.lstm = nn.LSTM(conv_filters*2, lstm_units, batch_first=True, bidirectional=True)
        
        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_units*2, lstm_units),
            nn.Tanh(),
            nn.Linear(lstm_units, 1)
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_units*2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        
        # CNN Feature Extraction
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # (batch_size, new_seq_len, conv_filters*2)
        
        # LSTM Temporal Processing
        lstm_out, _ = self.lstm(x)
        
        # Attention Mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.fc(context_vector)
        return output

# Example usage:
# model = TimeSeriesClassifier(input_size=13, num_classes=2, seq_len=60)
# model = model.to(device)
