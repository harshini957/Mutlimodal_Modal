import os
import json
import pandas as pd
from PIL import Image
import numpy as np


def create_annotations_from_folders(merged_dir, normal_dir, output_file):
    """Create annotations file from folder structure"""
    annotations = {}

    # Process merged dataset (should have subdirectories for benign/malignant)
    if os.path.exists(merged_dir):
        for subdir in os.listdir(merged_dir):
            subdir_path = os.path.join(merged_dir, subdir)
            if os.path.isdir(subdir_path) and subdir in ['benign', 'malignant']:
                for img_name in os.listdir(subdir_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # For now, create full image bounding box
                        img_path = os.path.join(subdir_path, img_name)
                        img = Image.open(img_path)
                        w, h = img.size

                        annotations[img_name] = {
                            'objects': [{
                                'label': subdir,
                                'bbox': [0, 0, w, h]  # Full image bbox
                            }]
                        }

    # Save annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Created annotations for {len(annotations)} images")
    return annotations


def verify_texture_features(texture_file, image_dirs):
    """Verify that texture features exist for all images"""
    texture_df = pd.read_excel(texture_file)
    texture_names = set(texture_df.iloc[:, 0].values)

    missing_features = []

    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            for root, dirs, files in os.walk(img_dir):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        base_name = os.path.splitext(file)[0]
                        if file not in texture_names and base_name not in texture_names:
                            missing_features.append(file)

    if missing_features:
        print(f"Warning: Missing texture features for {len(missing_features)} images:")
        for img in missing_features[:10]:  # Show first 10
            print(f"  - {img}")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")
    else:
        print("All images have corresponding texture features")

    return missing_features