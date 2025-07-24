import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class DepthAnythingV2Encoder(nn.Module):
    """Real Depth-Anything v2 encoder for depth maps"""
    
    def __init__(self, model_name: str = "LiheYoung/depth_anything_vitl14", 
                 feature_dim: int = 1024, output_dim: int = 256, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        try:
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            print(f"✅ Loaded Depth-Anything v2: {model_name}")
        except ImportError:
            print("⚠️  Warning: transformers not available, using CNN fallback")
            self.processor = None
            self.model = None
        
        # Feature projection to output dimension
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Fallback CNN for depth encoding
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # CNN feature projection
        self.cnn_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def preprocess_depth(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Preprocess depth maps for the model"""
        # Ensure depth maps are in the right format
        if depth_maps.dim() == 3:
            depth_maps = depth_maps.unsqueeze(1)  # Add channel dimension
        
        # Normalize depth values to [0, 1]
        depth_min = depth_maps.min()
        depth_max = depth_maps.max()
        if depth_max > depth_min:
            depth_maps = (depth_maps - depth_min) / (depth_max - depth_min)
        
        # Resize to model input size (typically 224x224)
        if depth_maps.shape[-1] != 224 or depth_maps.shape[-2] != 224:
            depth_maps = F.interpolate(depth_maps, size=(224, 224), mode='bilinear', align_corners=False)
        
        return depth_maps
    
    def encode_depth(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Encode depth maps using Depth-Anything v2"""
        if self.model is not None:
            try:
                # Preprocess depth maps
                processed_depth = self.preprocess_depth(depth_maps)
                
                # Convert to PIL images for the processor
                batch_size = processed_depth.shape[0]
                pil_images = []
                
                for i in range(batch_size):
                    depth_img = processed_depth[i, 0].cpu().numpy()
                    depth_img = (depth_img * 255).astype(np.uint8)
                    from PIL import Image
                    pil_img = Image.fromarray(depth_img, mode='L')
                    pil_images.append(pil_img)
                
                # Process with Depth-Anything
                inputs = self.processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence length
                
                # Project to output dimension
                encoded_features = self.feature_projection(features)
                return encoded_features
                
            except Exception as e:
                print(f"⚠️  Depth-Anything encoding failed: {e}, using fallback")
                return self._fallback_depth_encoding(depth_maps)
        else:
            return self._fallback_depth_encoding(depth_maps)
    
    def _fallback_depth_encoding(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Fallback CNN encoding for depth maps"""
        # Preprocess depth maps
        processed_depth = self.preprocess_depth(depth_maps)
        
        # Encode with CNN
        cnn_features = self.depth_cnn(processed_depth)
        
        # Project to output dimension
        encoded_features = self.cnn_projection(cnn_features)
        
        return encoded_features
    
    def forward(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.encode_depth(depth_maps)
