# inference.py - Inference script
import torch
import yaml
from eval import MultimodalEvaluator
from PIL import Image
import pandas as pd
import numpy as np


def run_inference():
    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize evaluator
    evaluator = MultimodalEvaluator("multimodal_detector.pth", config)

    # Example inference
    image_path = "path/to/test/image.jpg"
    texture_features = np.random.rand(config['texture_input_dim'])  # Replace with actual features

    # Make prediction
    result = evaluator.predict(image_path, texture_features)

    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Bounding box: {result['bbox']}")
    print(f"Class probabilities: {result['class_probabilities']}")


if __name__ == "__main__":
    run_inference()
