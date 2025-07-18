import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Regression head for bounding box prediction"""

    def __init__(self, input_dim=2048, hidden_dim=512):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 4)  # x, y, w, h
        )

    def forward(self, features):
        return self.regressor(features)
