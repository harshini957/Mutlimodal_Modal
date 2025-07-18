import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from .Texture_loader import TextureFeatureLoader


class MultimodalDataset(Dataset):
    """Dataset for multimodal detection"""

    def __init__(self, merged_images_dir, normal_images_dir, annotations_file,
                 texture_features_file, transform=None, mode='train'):

        self.merged_images_dir = merged_images_dir
        self.normal_images_dir = normal_images_dir
        self.transform = transform
        self.mode = mode

        # Load texture features
        self.texture_loader = TextureFeatureLoader(texture_features_file, normalize=True)

        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # Create image list
        self.image_list = []

        # Add merged dataset images (benign and malignant)
        for subdir in ['benign', 'malignant']:
            subdir_path = os.path.join(merged_images_dir, subdir)
            if os.path.exists(subdir_path):
                for img_name in os.listdir(subdir_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.image_list.append({
                            'path': os.path.join(subdir_path, img_name),
                            'name': img_name,
                            'type': 'merged',
                            'label': subdir
                        })

        # Add normal images
        if os.path.exists(normal_images_dir):
            for img_name in os.listdir(normal_images_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_list.append({
                        'path': os.path.join(normal_images_dir, img_name),
                        'name': img_name,
                        'type': 'normal',
                        'label': 'normal'
                    })

        # Create label mapping
        self.label_map = {'benign': 0, 'malignant': 1, 'normal': 2}

        print(f"Dataset loaded with {len(self.image_list)} images")
        print(f"Texture feature dimension: {self.texture_loader.get_feature_dimension()}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_info = self.image_list[idx]

        # Load image
        image = Image.open(img_info['path']).convert('RGB')
        original_size = image.size

        # Get texture features using the loader
        texture_features = torch.tensor(
            self.texture_loader.get_texture_features(img_info['name']),
            dtype=torch.float32
        )

        # Get annotations and create target
        if img_info['type'] == 'normal':
            # Normal images - full image as bounding box
            target = {
                'boxes': torch.tensor([[0, 0, original_size[0], original_size[1]]], dtype=torch.float32),
                'labels': torch.tensor([self.label_map['normal']], dtype=torch.long),
                'image_id': torch.tensor([idx]),
                'area': torch.tensor([original_size[0] * original_size[1]], dtype=torch.float32),
                'iscrowd': torch.tensor([0], dtype=torch.int64)
            }
        else:
            # Get annotation for merged dataset
            annotation = self.annotations.get(img_info['name'], {})

            if 'objects' in annotation:
                boxes = []
                labels = []
                areas = []

                for obj in annotation['objects']:
                    bbox = obj['bbox']  # [x, y, width, height]
                    x1, y1, w, h = bbox
                    x2, y2 = x1 + w, y1 + h

                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, original_size[0]))
                    y1 = max(0, min(y1, original_size[1]))
                    x2 = max(x1, min(x2, original_size[0]))
                    y2 = max(y1, min(y2, original_size[1]))

                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.label_map[obj['label']])
                    areas.append((x2 - x1) * (y2 - y1))

                target = {
                    'boxes': torch.tensor(boxes, dtype=torch.float32),
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'image_id': torch.tensor([idx]),
                    'area': torch.tensor(areas, dtype=torch.float32),
                    'iscrowd': torch.tensor([0] * len(boxes), dtype=torch.int64)
                }
            else:
                # Default target if no annotation found - use image label
                target = {
                    'boxes': torch.tensor([[0, 0, original_size[0], original_size[1]]], dtype=torch.float32),
                    'labels': torch.tensor([self.label_map[img_info['label']]], dtype=torch.long),
                    'image_id': torch.tensor([idx]),
                    'area': torch.tensor([original_size[0] * original_size[1]], dtype=torch.float32),
                    'iscrowd': torch.tensor([0], dtype=torch.int64)
                }

        if self.transform:
            image = self.transform(image)

        return image, texture_features, target


def collate_fn(batch):
    """Custom collate function for multimodal data"""
    images, texture_features, targets = zip(*batch)

    # Stack texture features
    texture_features = torch.stack(texture_features)

    return list(images), texture_features, list(targets)
