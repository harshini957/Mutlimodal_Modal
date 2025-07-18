import torch
import torch.nn as nn


class TextureFeatureEncoder(nn.Module):
    """MLP to encode texture features to match image feature dimension"""

    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=1024, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final layer to match image feature dimension
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, texture_features):
        return self.mlp(texture_features)