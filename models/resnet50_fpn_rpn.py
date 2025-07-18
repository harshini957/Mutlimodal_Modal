import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign


class ImageFeatureExtractor(nn.Module):
    """ResNet-50 + FPN + RPN pipeline for image feature extraction"""

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        # Create backbone with FPN
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)

        # Define anchor generator
        # anchor_generator = AnchorGenerator(
        #     sizes=((32, 64, 128, 256, 512),),
        #     aspect_ratios=((0.5, 1.0, 2.0),) * 5
        # )

        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        # ROI pooling
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # Create Faster R-CNN but we'll extract features before final heads
        self.faster_rcnn = FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        # Feature dimension after ROI pooling
        # self.feature_dim = 256 * 7 * 7  # 12544
        self.feature_dim = 1024

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("targets should not be None during training")

        if isinstance(images.tensors, list):
            images.tensors = torch.stack(images.tensors)

        # Extract features from backbone
        features = self.backbone(images.tensors)

        # Generate proposals using RPN
        proposals, proposal_losses = self.faster_rcnn.rpn(images, features, targets)

        # Apply ROI pooling to get region features
        # if self.training:
        #     # During training, use ground truth boxes as well
        #     proposals = self.faster_rcnn.roi_heads.select_training_samples(proposals, targets)
        #     box_features = self.faster_rcnn.roi_heads.box_roi_pool(features, proposals[0].proposal_boxes,
        #                                                            images.image_sizes)
        # else:
        #     box_features = self.faster_rcnn.roi_heads.box_roi_pool(features, proposals, images.image_sizes)

        if self.training:
            # During training, use ground truth boxes as well
            proposals, _, _, _ = self.faster_rcnn.roi_heads.select_training_samples(proposals, targets)

            # Extract boxes
            # boxes = [p.bbox for p in proposals]
            boxes = proposals
            box_features = self.faster_rcnn.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
        else:
            box_features = self.faster_rcnn.roi_heads.box_roi_pool(features, proposals, images.image_sizes)

        # Flatten features
        box_features = self.faster_rcnn.roi_heads.box_head(box_features)
        print("Image features shape:", box_features.shape)

        return box_features, proposals, proposal_losses