import torch
import cv2
import numpy as np
from models.multimodal_model import MultimodalDetector
from utils.dataloader import MultimodalDataset
from torchvision import transforms
from torchvision.models.detection.image_list import ImageList


class MultimodalEvaluator:
    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # Load model
        self.model = MultimodalDetector(
            texture_input_dim=config['texture_input_dim'],
            num_classes=config['num_classes'],
            image_freeze=False
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Class names
        self.class_names = ['benign', 'malignant', 'normal']

    def predict(self, image_path, texture_features):
        """Make prediction on a single image"""
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Prepare texture features
        texture_tensor = torch.tensor(texture_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Create ImageList
        image_list = ImageList([image_tensor.squeeze(0)], [(image_tensor.shape[-2], image_tensor.shape[-1])])

        with torch.no_grad():
            outputs = self.model(image_list, texture_tensor)

            # Get predictions
            class_probs = torch.softmax(outputs['class_logits'], dim=1)
            predicted_class = torch.argmax(class_probs, dim=1).item()
            confidence = class_probs[0, predicted_class].item()

            bbox = outputs['bbox_preds'][0].cpu().numpy()

        return {
            'class': self.class_names[predicted_class],
            'confidence': confidence,
            'bbox': bbox,
            'class_probabilities': class_probs[0].cpu().numpy()
        }