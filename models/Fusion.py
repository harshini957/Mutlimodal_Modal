import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """Fusion block to combine image and texture features"""

    def __init__(self, image_dim=1024, texture_dim=1024, fusion_dim=2048):
        super().__init__()

        self.image_dim = image_dim
        self.texture_dim = texture_dim
        self.fusion_dim = fusion_dim

        # Projection layers
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.3)
        )

        self.texture_proj = nn.Sequential(
            nn.Linear(texture_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.3)
        )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.3)
        )

    def forward(self, image_features, texture_features):
        # Project features
        img_proj = self.image_proj(image_features)
        tex_proj = self.texture_proj(texture_features)

        # Concatenate and fuse
        combined = torch.cat([img_proj, tex_proj], dim=1)
        fused_features = self.fusion_layer(combined)

        return fused_features