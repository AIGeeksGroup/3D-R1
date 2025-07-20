import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class SigLIP2ImageEncoder(nn.Module):
    """
    Image encoder using SigLIP-2 (ViT-L/14)
    """
    
    def __init__(self, 
                 model_name: str = "google/siglip-vit-large-patch14-384",
                 feature_dim: int = 1024,
                 output_dim: int = 256,
                 device: str = "cuda"):
        super().__init__()
        
        self.device = device
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # Load SigLIP-2 model
        try:
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
        except ImportError:
            print("Warning: transformers not available, using dummy image encoder")
            self.processor = None
            self.model = None
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Image-specific processing (fallback)
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for SigLIP-2
        
        Args:
            images: (B, 3, H, W) RGB images
            
        Returns:
            Preprocessed images
        """
        # Normalize to [0, 1] if needed
        if images.max() > 1.0:
            images = images / 255.0
        
        # SigLIP-2 expects specific normalization
        # This is a simplified version - in practice you'd use the processor
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(images.device)
        images = (images - mean) / std
        
        return images
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using SigLIP-2
        
        Args:
            images: (B, num_views, 3, H, W) RGB images
            
        Returns:
            Image features (B, output_dim)
        """
        if self.model is None:
            # Fallback to simple image processing
            return self._fallback_image_encoding(images)
        
        batch_size = images.shape[0]
        num_views = images.shape[1]  # Number of images per sample
        features_list = []
        
        for i in range(batch_size):
            batch_features = []
            for j in range(num_views):
                # Process each image individually
                single_image = images[i, j].unsqueeze(0)  # Add batch dimension
                processed_image = self.preprocess_images(single_image)
                
                # Process with SigLIP-2
                with torch.no_grad():
                    # Use the model's forward pass to get features
                    outputs = self.model(processed_image, output_hidden_states=True)
                    
                    # Extract features from the last layer
                    if hasattr(outputs, 'last_hidden_state'):
                        features = outputs.last_hidden_state
                    elif hasattr(outputs, 'pooler_output'):
                        features = outputs.pooler_output
                    else:
                        # Fallback: use the first output
                        features = outputs[0]
                    
                    # Global average pooling if needed
                    if features.dim() > 2:
                        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
                    
                    # Project to output dimension
                    features = self.feature_projection(features)
                    
                    batch_features.append(features)
            
            # Average features across all views for this sample
            if batch_features:
                avg_features = torch.mean(torch.stack(batch_features), dim=0)
                features_list.append(avg_features)
            else:
                features_list.append(torch.zeros(1, self.output_dim))
        
        return torch.cat(features_list, dim=0)
    
    def _fallback_image_encoding(self, images: torch.Tensor) -> torch.Tensor:
        """
        Fallback image encoding when SigLIP-2 is not available
        """
        batch_size = images.shape[0]
        num_views = images.shape[1]  # Number of images per sample
        features_list = []
        
        for i in range(batch_size):
            batch_features = []
            for j in range(num_views):
                # Process each image individually
                single_image = images[i, j].unsqueeze(0)  # Add batch dimension
                
                # Simple image feature extraction
                features = self.image_conv(single_image)
                features = features.squeeze(-1).squeeze(-1)  # (1, 256)
                
                # Project to output dimension
                features = self.feature_projection(features)
                
                batch_features.append(features)
            
            # Average features across all views for this sample
            if batch_features:
                avg_features = torch.mean(torch.stack(batch_features), dim=0)
                features_list.append(avg_features)
            else:
                features_list.append(torch.zeros(1, self.output_dim))
        
        return torch.cat(features_list, dim=0)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image encoding
        
        Args:
            images: (B, 3, H, W) RGB images
            
        Returns:
            Encoded image features (B, output_dim)
        """
        return self.encode_images(images)
