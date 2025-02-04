import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        
        self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=10, padding='same', bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=20, padding='same', bias=False)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=40, padding='same', bias=False)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Bottleneck layer
        x = self.bottleneck(x)
        
        # Parallel convolutions
        branch1 = self.conv1(x)
        branch2 = self.conv2(x)
        branch3 = self.conv3(x)
        
        # Max pooling branch
        branch4 = self.maxpool(x)
        branch4 = self.conv4(branch4)
        
        # Concatenate all branches
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        # Batch normalization and ReLU
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class InceptionTime(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, depth=6):
        super(InceptionTime, self).__init__()
        
        self.inception_modules = nn.ModuleList()
        for i in range(depth):
            self.inception_modules.append(InceptionModule(in_channels if i == 0 else out_channels * 4, out_channels))
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels * 4, num_classes)
        
    def forward(self, x):
        for module in self.inception_modules:
            x = module(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = nn.Sigmoid()(x)
        
        return x

# Example usage
if __name__ == "__main__":
    model = InceptionTime(in_channels=13, out_channels=32, num_classes=1, depth=6)
    x = torch.randn(32, 13, 60)  # Batch size of 32, 1 channel, sequence length of 128
    output = model(x)
    print(output.shape) 