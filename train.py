import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
import numpy as np
from models.multi_modal import MultimodalDetector
from utils.Dataloader import MultimodalDataset, collate_fn
from torchvision import transforms
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou


class MultimodalTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_data()
        self.setup_optimizers()

    def setup_model(self):
        """Initialize the multimodal model"""
        self.model = MultimodalDetector(
            texture_input_dim=self.config['texture_input_dim'],
            num_classes=self.config['num_classes'],
            image_freeze=True
        ).to(self.device)

    def setup_data(self):
        """Setup data loaders"""
        transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = MultimodalDataset(
            merged_images_dir=self.config['merged_images_dir'],
            normal_images_dir=self.config['normal_images_dir'],
            annotations_file=self.config['annotations_file'],
            texture_features_file=self.config['texture_features_file'],
            transform=transform,
            mode='train'
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )

    def setup_optimizers(self):
        """Setup optimizers with different learning rates"""
        # Texture pipeline optimizer
        texture_params = list(self.model.texture_encoder.parameters()) + \
                         list(self.model.fusion_block.parameters()) + \
                         list(self.model.classification_head.parameters()) + \
                         list(self.model.regression_head.parameters())

        self.texture_optimizer = optim.AdamW(
            texture_params,
            lr=self.config['texture_lr'],
            weight_decay=self.config['weight_decay']
        )

        # Image pipeline optimizer (will be used in phase 2)
        self.image_optimizer = optim.AdamW(
            self.model.image_extractor.parameters(),
            lr=self.config['image_lr'],
            weight_decay=self.config['weight_decay']
        )

        # Combined optimizer (will be used in phase 2)
        self.combined_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['combined_lr'],
            weight_decay=self.config['weight_decay']
        )

        # Schedulers
        self.texture_scheduler = optim.lr_scheduler.StepLR(
            self.texture_optimizer,
            step_size=self.config['scheduler_step_size'],
            gamma=self.config['scheduler_gamma']
        )

        self.image_scheduler = optim.lr_scheduler.StepLR(
            self.image_optimizer,
            step_size=self.config['scheduler_step_size'],
            gamma=self.config['scheduler_gamma']
        )

        self.combined_scheduler = optim.lr_scheduler.StepLR(
            self.combined_optimizer,
            step_size=self.config['scheduler_step_size'],
            gamma=self.config['scheduler_gamma']
        )

    def assign_targets_to_proposals(self, proposals, targets, iou_threshold=0.5):
        """
        Assign ground truth labels to proposals based on IoU overlap
        """
        assigned_labels = []
        assigned_boxes = []

        for i, (props, target) in enumerate(zip(proposals, targets)):
            if len(props) == 0:
                continue

            gt_boxes = target['boxes']
            gt_labels = target['labels']

            # Calculate IoU between proposals and ground truth boxes
            if len(gt_boxes) > 0:
                ious = box_iou(props, gt_boxes)
                max_ious, max_indices = ious.max(dim=1)

                # Assign labels based on IoU threshold
                proposal_labels = torch.zeros(len(props), dtype=torch.long, device=props.device)
                proposal_boxes = torch.zeros_like(props)

                # Positive samples (IoU > threshold)
                positive_mask = max_ious > iou_threshold
                proposal_labels[positive_mask] = gt_labels[max_indices[positive_mask]]
                proposal_boxes[positive_mask] = gt_boxes[max_indices[positive_mask]]

                # Background samples (IoU <= threshold)
                proposal_labels[~positive_mask] = 0  # Background class
                proposal_boxes[~positive_mask] = props[~positive_mask]  # Keep proposal box

            else:
                # No ground truth boxes, all proposals are background
                proposal_labels = torch.zeros(len(props), dtype=torch.long, device=props.device)
                proposal_boxes = props

            assigned_labels.append(proposal_labels)
            assigned_boxes.append(proposal_boxes)

        return assigned_labels, assigned_boxes

    def train_phase1(self):
        """Phase 1: Train only texture pipeline with frozen image pipeline"""
        print("Phase 1: Training texture pipeline only...")

        self.model.freeze_image_pipeline()
        criterion_cls = nn.CrossEntropyLoss()
        criterion_bbox = nn.SmoothL1Loss()

        for epoch in range(self.config['phase1_epochs']):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, (images, texture_features, targets) in enumerate(tqdm(self.train_loader)):
                try:
                    # Move to device
                    images = [img.to(self.device) for img in images]
                    texture_features = texture_features.to(self.device)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # Create ImageList for Faster R-CNN
                    image_list = ImageList(images, [(img.shape[-2], img.shape[-1]) for img in images])

                    # Forward pass
                    outputs = self.model(image_list, texture_features, targets)

                    # Skip if no proposals
                    if outputs['class_logits'].size(0) == 0:
                        continue

                    # Assign targets to proposals
                    assigned_labels, assigned_boxes = self.assign_targets_to_proposals(
                        outputs['proposals'], targets
                    )

                    # Skip if no assignments
                    if not assigned_labels:
                        continue

                    # Concatenate all assigned labels and boxes
                    all_labels = torch.cat(assigned_labels)
                    all_boxes = torch.cat(assigned_boxes)

                    # Ensure dimensions match
                    if outputs['class_logits'].size(0) != all_labels.size(0):
                        min_size = min(outputs['class_logits'].size(0), all_labels.size(0))
                        outputs['class_logits'] = outputs['class_logits'][:min_size]
                        outputs['bbox_preds'] = outputs['bbox_preds'][:min_size]
                        all_labels = all_labels[:min_size]
                        all_boxes = all_boxes[:min_size]

                    # Calculate losses only for positive samples (non-background)
                    positive_mask = all_labels > 0

                    if positive_mask.sum() > 0:
                        cls_loss = criterion_cls(outputs['class_logits'], all_labels)
                        bbox_loss = criterion_bbox(
                            outputs['bbox_preds'][positive_mask],
                            all_boxes[positive_mask]
                        )
                    else:
                        cls_loss = criterion_cls(outputs['class_logits'], all_labels)
                        bbox_loss = torch.tensor(0.0, device=self.device)

                    # RPN loss
                    rpn_loss = sum(loss for loss in outputs['proposal_losses'].values()) if outputs[
                        'proposal_losses'] else torch.tensor(0.0, device=self.device)

                    total_loss_batch = cls_loss + bbox_loss + rpn_loss

                    # Backward pass
                    self.texture_optimizer.zero_grad()
                    total_loss_batch.backward()
                    self.texture_optimizer.step()

                    total_loss += total_loss_batch.item()
                    num_batches += 1

                    if batch_idx % 10 == 0:
                        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}')
                        print(
                            f'  Class Loss: {cls_loss.item():.4f}, BBox Loss: {bbox_loss.item():.4f}, RPN Loss: {rpn_loss.item():.4f}')
                        print(f'  Predictions: {outputs["class_logits"].size(0)}, Targets: {all_labels.size(0)}')

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue

            if num_batches > 0:
                self.texture_scheduler.step()
                print(f'Phase 1 - Epoch {epoch}, Average Loss: {total_loss / num_batches:.4f}')
            else:
                print(f'Phase 1 - Epoch {epoch}, No valid batches processed')

    def train_phase2(self):
        """Phase 2: Train both pipelines together"""
        print("Phase 2: Training both pipelines together...")

        self.model.unfreeze_image_pipeline()
        criterion_cls = nn.CrossEntropyLoss()
        criterion_bbox = nn.SmoothL1Loss()

        for epoch in range(self.config['phase2_epochs']):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, (images, texture_features, targets) in enumerate(tqdm(self.train_loader)):
                try:
                    # Move to device
                    images = [img.to(self.device) for img in images]
                    texture_features = texture_features.to(self.device)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # Create ImageList for Faster R-CNN
                    image_list = ImageList(images, [(img.shape[-2], img.shape[-1]) for img in images])

                    # Forward pass
                    outputs = self.model(image_list, texture_features, targets)

                    # Skip if no proposals
                    if outputs['class_logits'].size(0) == 0:
                        continue

                    # Assign targets to proposals
                    assigned_labels, assigned_boxes = self.assign_targets_to_proposals(
                        outputs['proposals'], targets
                    )

                    # Skip if no assignments
                    if not assigned_labels:
                        continue

                    # Concatenate all assigned labels and boxes
                    all_labels = torch.cat(assigned_labels)
                    all_boxes = torch.cat(assigned_boxes)

                    # Ensure dimensions match
                    if outputs['class_logits'].size(0) != all_labels.size(0):
                        min_size = min(outputs['class_logits'].size(0), all_labels.size(0))
                        outputs['class_logits'] = outputs['class_logits'][:min_size]
                        outputs['bbox_preds'] = outputs['bbox_preds'][:min_size]
                        all_labels = all_labels[:min_size]
                        all_boxes = all_boxes[:min_size]

                    # Calculate losses only for positive samples (non-background)
                    positive_mask = all_labels > 0

                    if positive_mask.sum() > 0:
                        cls_loss = criterion_cls(outputs['class_logits'], all_labels)
                        bbox_loss = criterion_bbox(
                            outputs['bbox_preds'][positive_mask],
                            all_boxes[positive_mask]
                        )
                    else:
                        cls_loss = criterion_cls(outputs['class_logits'], all_labels)
                        bbox_loss = torch.tensor(0.0, device=self.device)

                    # RPN loss
                    rpn_loss = sum(loss for loss in outputs['proposal_losses'].values()) if outputs[
                        'proposal_losses'] else torch.tensor(0.0, device=self.device)

                    total_loss_batch = cls_loss + bbox_loss + rpn_loss

                    # Backward pass
                    self.combined_optimizer.zero_grad()
                    total_loss_batch.backward()
                    self.combined_optimizer.step()

                    total_loss += total_loss_batch.item()
                    num_batches += 1

                    if batch_idx % 10 == 0:
                        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}')
                        print(
                            f'  Class Loss: {cls_loss.item():.4f}, BBox Loss: {bbox_loss.item():.4f}, RPN Loss: {rpn_loss.item():.4f}')

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue

            if num_batches > 0:
                self.combined_scheduler.step()
                print(f'Phase 2 - Epoch {epoch}, Average Loss: {total_loss / num_batches:.4f}')
            else:
                print(f'Phase 2 - Epoch {epoch}, No valid batches processed')

    def train(self):
        """Main training function"""
        # Phase 1: Train texture pipeline only
        self.train_phase1()

        # Phase 2: Train both pipelines
        self.train_phase2()

        # Save model
        torch.save(self.model.state_dict(), 'multimodal_detector.pth')
        print("Training completed and model saved!")