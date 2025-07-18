import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head for 3 classes: benign, malignant, normal"""

    def __init__(self, input_dim=2048, num_classes=3, hidden_dim=512):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features):
        return self.classifier(features)