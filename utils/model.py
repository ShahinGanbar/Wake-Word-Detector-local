# utils/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WakeWordCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (1, 64, 32)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)        # → (16, 32, 16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)        # → (32, 16, 8)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)        # → (64, 8, 4)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 4, 128)
        self.fc2 = nn.Linear(128, 2)        # binary classification

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
