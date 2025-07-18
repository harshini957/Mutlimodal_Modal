
# utils/texture_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import os


class TextureFeatureLoader:
    """Load and preprocess texture features from Excel file"""

    def __init__(self, excel_path, normalize=True):
        self.excel_path = excel_path
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None

        # Load texture features
        self.texture_df = pd.read_excel(excel_path,engine='openpyxl')

        # Get feature dimension (excluding image name column)
        self.feature_dim = self.texture_df.shape[1] - 1

        print(f"Loaded texture features with dimension: {self.feature_dim}")
        print(f"Number of samples: {len(self.texture_df)}")

        # Prepare features for normalization
        if self.normalize:
            # Assuming first column is image name, rest are features
            feature_columns = self.texture_df.columns[1:]
            feature_data = self.texture_df[feature_columns].values

            # Fit scaler on all features
            self.scaler.fit(feature_data)

            # Apply normalization to the dataframe
            self.texture_df[feature_columns] = self.scaler.transform(feature_data)

            print("Texture features normalized using StandardScaler")

    def get_texture_features(self, image_name):
        """Get texture features for a specific image"""
        # Remove file extension for matching
        image_name_base = os.path.splitext(image_name)[0]

        # Try exact match first
        row = self.texture_df[self.texture_df.iloc[:, 0] == image_name]

        # If not found, try without extension
        if len(row) == 0:
            row = self.texture_df[self.texture_df.iloc[:, 0] == image_name_base]

        # If still not found, try with common extensions
        if len(row) == 0:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                row = self.texture_df[self.texture_df.iloc[:, 0] == image_name_base + ext]
                if len(row) > 0:
                    break

        if len(row) == 0:
            print(f"Warning: No texture features found for {image_name}, using zeros")
            return np.zeros(self.feature_dim)

        # Return feature values (excluding image name column)
        return row.iloc[0, 1:].values.astype(np.float32)

    def get_feature_dimension(self):
        """Get the dimension of texture features"""
        return self.feature_dim

    def get_all_features(self):
        """Get all texture features as numpy array"""
        return self.texture_df.iloc[:, 1:].values.astype(np.float32)

    def get_all_image_names(self):
        """Get all image names"""
        return self.texture_df.iloc[:, 0].values.tolist()
