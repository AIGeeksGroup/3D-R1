import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import cv2

class DepthAnythingV2Encoder(nn.Module):
    """Depth-Anything-V2 encoder for depth maps based on official implementation"""
    
    def __init__(self, model_name: str = "LiheYoung/depth_anything_vitl14", 
                 feature_dim: int = 1024, output_dim: int = 256, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.model_name = model_name
        
        # Try to load Depth-Anything-V2 model using official method
        try:
            # First try to load using transformers pipeline (recommended by official repo)
            from transformers import pipeline
            try:
                # Map model names to official model names
                model_mapping = {
                    "LiheYoung/depth_anything_vitl14": "depth-anything/Depth-Anything-V2-Large-hf",
                    "LiheYoung/depth_anything_vitb14": "depth-anything/Depth-Anything-V2-Base-hf", 
                    "LiheYoung/depth_anything_vits14": "depth-anything/Depth-Anything-V2-Small-hf"
                }
                
                official_model_name = model_mapping.get(model_name, "depth-anything/Depth-Anything-V2-Large-hf")
                self.pipeline = pipeline(task="depth-estimation", model=official_model_name)
                self.model = None
                self.processor = None
                print(f"✅ Loaded Depth-Anything-V2 using transformers pipeline: {official_model_name}")
                
            except Exception as e:
                print(f"⚠️  Failed to load via transformers pipeline: {e}")
                print("⚠️  Trying direct model loading...")
                self.pipeline = None
                self._load_direct_model()
                
        except ImportError:
            print("⚠️  Warning: transformers not available, using CNN fallback")
            self.pipeline = None
            self.model = None
            self.processor = None
        
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
    
    def _load_direct_model(self):
        """Load Depth-Anything-V2 model directly using official method"""
        try:
            from transformers import AutoModel, AutoImageProcessor
            
            # Try loading with trust_remote_code and specific parameters
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Try to load processor, but don't fail if it doesn't work
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                print(f"✅ Loaded Depth-Anything-V2 model and processor: {self.model_name}")
            except Exception as e:
                print(f"⚠️  Processor loading failed: {e}, will use manual preprocessing")
                self.processor = None
                
        except Exception as e:
            print(f"⚠️  Direct model loading failed: {e}")
            print("⚠️  Using CNN fallback for depth encoding")
            self.model = None
            self.processor = None
    
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
        
        # Resize to model input size (typically 518 for Depth-Anything-V2)
        if depth_maps.shape[-1] != 518 or depth_maps.shape[-2] != 518:
            depth_maps = F.interpolate(depth_maps, size=(518, 518), mode='bilinear', align_corners=False)
        
        return depth_maps
    
    def encode_depth_pipeline(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Encode depth maps using transformers pipeline"""
        batch_size = depth_maps.shape[0]
        num_views = depth_maps.shape[1]  # Number of depth maps per sample
        features_list = []
        
        for i in range(batch_size):
            batch_features = []
            for j in range(num_views):
                # Convert depth map to PIL image
                depth_img = depth_maps[i, j].cpu().numpy()
                depth_img = (depth_img * 255).astype(np.uint8)
                
                # Convert to RGB (Depth-Anything expects RGB)
                depth_rgb = np.stack([depth_img] * 3, axis=-1)
                
                from PIL import Image
                pil_img = Image.fromarray(depth_rgb)
                
                # Use pipeline for inference
                with torch.no_grad():
                    result = self.pipeline(pil_img)
                    depth_pred = result["depth"]
                    
                    # Convert PIL depth to tensor and extract features
                    depth_tensor = torch.from_numpy(np.array(depth_pred)).float().unsqueeze(0).unsqueeze(0)
                    depth_tensor = F.interpolate(depth_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                    
                    # Use the depth prediction as features (simplified approach)
                    features = depth_tensor.flatten(1)  # Flatten spatial dimensions
                    
                    # Pad or truncate to expected feature dimension
                    if features.shape[1] < self.feature_dim:
                        padding = torch.zeros(1, self.feature_dim - features.shape[1])
                        features = torch.cat([features, padding], dim=1)
                    else:
                        features = features[:, :self.feature_dim]
                    
                    batch_features.append(features)
            
            # Average features across all views for this sample
            if batch_features:
                avg_features = torch.mean(torch.stack(batch_features), dim=0)
                features_list.append(avg_features)
            else:
                # Fallback if no valid depth maps
                features_list.append(torch.zeros(1, self.feature_dim))
        
        features = torch.cat(features_list, dim=0)
        return features
    
    def encode_depth_direct(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Encode depth maps using direct model inference"""
        try:
            batch_size = depth_maps.shape[0]
            num_views = depth_maps.shape[1]  # Number of depth maps per sample
            features_list = []
            
            for i in range(batch_size):
                batch_features = []
                for j in range(num_views):
                    # Process each depth map individually
                    single_depth = depth_maps[i, j].unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                    processed_depth = self.preprocess_depth(single_depth)
                    
                    # Convert to RGB format (Depth-Anything expects RGB)
                    rgb_depth = processed_depth.repeat(1, 3, 1, 1)
                    
                    # Normalize to ImageNet stats
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                    rgb_depth = (rgb_depth - mean) / std
                    
                    with torch.no_grad():
                        outputs = self.model(rgb_depth)
                        if hasattr(outputs, 'last_hidden_state'):
                            features = outputs.last_hidden_state.mean(dim=1)
                        else:
                            features = outputs[0].mean(dim=1)  # Assume first output is hidden states
                        
                        # Ensure features have the right dimension
                        if features.shape[1] != self.feature_dim:
                            # Project to expected dimension
                            if hasattr(self, 'feature_adapter'):
                                features = self.feature_adapter(features)
                            else:
                                # Create a simple adapter if needed
                                self.feature_adapter = nn.Linear(features.shape[1], self.feature_dim).to(self.device)
                                features = self.feature_adapter(features)
                        
                        batch_features.append(features)
                
                # Average features across all views for this sample
                if batch_features:
                    avg_features = torch.mean(torch.stack(batch_features), dim=0)
                    features_list.append(avg_features)
                else:
                    features_list.append(torch.zeros(1, self.feature_dim))
            
            return torch.cat(features_list, dim=0)
            
        except Exception as e:
            print(f"⚠️  Direct model inference failed: {e}, using CNN fallback")
            return self._fallback_depth_encoding(depth_maps)
    
    def encode_depth(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Encode depth maps using the best available method"""
        if self.pipeline is not None:
            # Use transformers pipeline (recommended by official repo)
            features = self.encode_depth_pipeline(depth_maps)
        elif self.model is not None:
            # Use direct model inference
            features = self.encode_depth_direct(depth_maps)
        else:
            # Use CNN fallback
            features = self._fallback_depth_encoding(depth_maps)
        
        # Project to output dimension
        encoded_features = self.feature_projection(features)
        return encoded_features
    
    def _fallback_depth_encoding(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Fallback CNN encoding for depth maps"""
        batch_size = depth_maps.shape[0]
        num_views = depth_maps.shape[1]  # Number of depth maps per sample
        features_list = []
        
        for i in range(batch_size):
            batch_features = []
            for j in range(num_views):
                # Process each depth map individually
                single_depth = depth_maps[i, j].unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                processed_depth = self.preprocess_depth(single_depth)
                
                # Encode with CNN
                cnn_features = self.depth_cnn(processed_depth)
                
                # Project to output dimension
                encoded_features = self.cnn_projection(cnn_features)
                
                batch_features.append(encoded_features)
            
            # Average features across all views for this sample
            if batch_features:
                avg_features = torch.mean(torch.stack(batch_features), dim=0)
                features_list.append(avg_features)
            else:
                features_list.append(torch.zeros(1, self.output_dim))
        
        return torch.cat(features_list, dim=0)
    
    def forward(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.encode_depth(depth_maps)
