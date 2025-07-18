import torch
import torch.nn as nn
from .resnet50_fpn_rpn import ImageFeatureExtractor
from .mlp_texture import TextureFeatureEncoder
from .Fusion import FusionBlock
from .Classification_head import ClassificationHead
from .Regression_head import RegressionHead


class MultimodalDetector(nn.Module):
    """Complete multimodal detection system"""

    def __init__(self, texture_input_dim, num_classes=3, image_freeze=True):
        super().__init__()

        # Image processing pipeline
        self.image_extractor = ImageFeatureExtractor(num_classes=num_classes)
        self.image_feature_dim = self.image_extractor.feature_dim  # 12544

        # Texture processing pipeline - match image feature dimension
        self.texture_encoder = TextureFeatureEncoder(
            input_dim=texture_input_dim,
            output_dim=self.image_feature_dim  # Match exactly
        )

        # Fusion block
        self.fusion_block = FusionBlock(
            image_dim=self.image_feature_dim,
            texture_dim=self.image_feature_dim,
            fusion_dim=2048
        )

        # Output heads
        self.classification_head = ClassificationHead(
            input_dim=2048,
            num_classes=num_classes
        )

        self.regression_head = RegressionHead(input_dim=2048)

        # Freeze image pipeline initially
        if image_freeze:
            self.freeze_image_pipeline()

        print(f"Model initialized:")
        print(f"  - Image feature dimension: {self.image_feature_dim}")
        print(f"  - Texture input dimension: {texture_input_dim}")
        print(f"  - Texture output dimension: {self.image_feature_dim}")
        print(f"  - Fusion dimension: 2048")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - Image pipeline frozen: {image_freeze}")

    def freeze_image_pipeline(self):
        """Freeze image processing pipeline"""
        for param in self.image_extractor.parameters():
            param.requires_grad = False
        print("Image pipeline frozen")

    def unfreeze_image_pipeline(self):
        """Unfreeze image processing pipeline"""
        for param in self.image_extractor.parameters():
            param.requires_grad = True
        print("Image pipeline unfrozen")

    def get_trainable_parameters(self):
        """Get trainable parameters for different components"""
        image_params = [p for p in self.image_extractor.parameters() if p.requires_grad]
        texture_params = list(self.texture_encoder.parameters())
        fusion_params = list(self.fusion_block.parameters())
        cls_params = list(self.classification_head.parameters())
        reg_params = list(self.regression_head.parameters())

        non_image_params = texture_params + fusion_params + cls_params + reg_params

        return {
            'image_params': image_params,
            'texture_params': texture_params,
            'fusion_params': fusion_params,
            'classification_params': cls_params,
            'regression_params': reg_params,
            'non_image_params': non_image_params,
            'all_params': image_params + non_image_params
        }

    def forward(self, images, texture_features, targets=None):
        batch_size = texture_features.size(0)

        # Extract image features
        image_features, proposals, proposal_losses = self.image_extractor(images, targets)

        # Process texture features to match image feature dimension
        texture_features_encoded = self.texture_encoder(texture_features)

        # Handle different numbers of proposals per image
        if len(proposals) > 0:
            # Get number of proposals per image
            if hasattr(proposals[0], 'bbox'):
                # Training mode - proposals is a list of BoxList objects
                num_proposals_per_image = [len(prop.bbox) for prop in proposals]
            else:
                # Inference mode - proposals is a list of tensors
                num_proposals_per_image = [len(prop) for prop in proposals]

            # Expand texture features to match number of proposals
            texture_features_expanded = []
            for i, num_props in enumerate(num_proposals_per_image):
                if num_props > 0:
                    texture_expanded = texture_features_encoded[i:i + 1].repeat(num_props, 1)
                    texture_features_expanded.append(texture_expanded)

            if texture_features_expanded:
                texture_features_final = torch.cat(texture_features_expanded, dim=0)
            else:
                # No proposals, use original texture features
                texture_features_final = texture_features_encoded
        else:
            # No proposals, use original texture features
            texture_features_final = texture_features_encoded

        # Ensure matching dimensions
        if image_features.size(0) != texture_features_final.size(0):
            # Adjust texture features to match image features
            if image_features.size(0) > texture_features_final.size(0):
                repeat_factor = image_features.size(0) // texture_features_final.size(0)
                texture_features_final = texture_features_final.repeat(repeat_factor, 1)
            else:
                texture_features_final = texture_features_final[:image_features.size(0)]

        # Fuse features
        fused_features = self.fusion_block(image_features, texture_features_final)

        # Get predictions
        class_logits = self.classification_head(fused_features)
        bbox_preds = self.regression_head(fused_features)

        return {
            'class_logits': class_logits,
            'bbox_preds': bbox_preds,
            'proposals': proposals,
            'proposal_losses': proposal_losses,
            'image_features': image_features,
            'texture_features': texture_features_final,
            'fused_features': fused_features
        }