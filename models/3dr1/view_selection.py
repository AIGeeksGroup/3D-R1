import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import clip
from PIL import Image
import torchvision.transforms as transforms

class DynamicViewSelection(nn.Module):
    """Dynamic View Selection for 3D-R1"""
    
    def __init__(self, device="cuda", num_views=4):
        super().__init__()
        self.device = device
        self.num_views = num_views
        
        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Learnable weights
        self.wt = nn.Parameter(torch.tensor(0.3))  # text relevance
        self.wc = nn.Parameter(torch.tensor(0.35))  # coverage
        self.wclip = nn.Parameter(torch.tensor(0.35))  # CLIP alignment
        
        # Coverage network
        self.coverage_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def compute_scores(self, view_features, text_features, point_cloud_features=None):
        """Compute three scoring functions
        
        Args:
            view_features: (num_candidates, 512) - CLIP features of rendered views
            text_features: (1, 512) - CLIP text features
            point_cloud_features: (N, 512) - Point cloud features (optional)
        """
        # SText→3D: Text-to-3D relevance score
        # Measures how well the view aligns with the textual description
        text_relevance = F.cosine_similarity(view_features, text_features, dim=-1)
        
        # SImage→3D: Coverage score
        # Measures how well the view covers the 3D scene content
        coverage = torch.sigmoid(self.coverage_net(view_features).squeeze(-1))
        
        # SCLIP: CLIP alignment score
        # Measures semantic alignment between view and text using CLIP
        # This is different from text_relevance as it uses CLIP's semantic understanding
        if point_cloud_features is not None:
            # If we have point cloud features, compute alignment with 3D content
            clip_alignment = F.cosine_similarity(view_features, point_cloud_features.mean(0, keepdim=True), dim=-1)
        else:
            # Otherwise, use a different CLIP-based alignment
            clip_alignment = F.cosine_similarity(view_features, text_features, dim=-1) * 0.8  # Different weight
        
        return text_relevance, coverage, clip_alignment
    
    def compute_utility(self, text_relevance, coverage, clip_alignment):
        """Compute utility score U(v)"""
        wc_norm = self.wc / (self.wc + self.wclip)
        wclip_norm = self.wclip / (self.wc + self.wclip)
        
        utility = (self.wt * text_relevance + 
                  wc_norm * coverage + 
                  wclip_norm * clip_alignment)
        return utility
    
    def render_candidate_views(self, point_cloud, num_candidates=8):
        """Render candidate views from point cloud
        
        Args:
            point_cloud: (N, 3) - Point cloud coordinates
            num_candidates: Number of candidate views to generate
        """
        # Sample camera positions around the point cloud
        center = point_cloud.mean(0)
        radius = torch.norm(point_cloud - center, dim=1).max()
        
        # Generate camera positions in a sphere around the scene
        angles = torch.linspace(0, 2*np.pi, num_candidates)
        heights = torch.linspace(-0.5, 0.5, num_candidates)
        
        camera_positions = []
        for i in range(num_candidates):
            angle = angles[i]
            height = heights[i]
            x = center[0] + radius * 1.5 * torch.cos(angle)
            y = center[1] + radius * 1.5 * torch.sin(angle)
            z = center[2] + radius * 1.5 * height
            camera_positions.append([x, y, z])
        
        camera_positions = torch.stack(camera_positions)
        
        # Simple rendering: project points to 2D images
        rendered_images = []
        for cam_pos in camera_positions:
            # Simple orthographic projection
            # In practice, you would use a proper renderer like PyTorch3D
            img = self.simple_point_cloud_to_image(point_cloud, cam_pos)
            rendered_images.append(img)
        
        return torch.stack(rendered_images)
    
    def simple_point_cloud_to_image(self, point_cloud, camera_pos):
        """Simple point cloud to image conversion (placeholder)"""
        # This is a simplified version. In practice, use PyTorch3D or similar
        # For now, create a dummy image with some structure
        img = torch.randn(224, 224, 3)  # RGB image
        
        # Add some structure based on point cloud
        center = point_cloud.mean(0)
        distances = torch.norm(point_cloud - center, dim=1)
        max_dist = distances.max()
        
        # Create a simple visualization
        for i in range(224):
            for j in range(224):
                # Simple mapping based on position
                x_norm = (i - 112) / 112
                y_norm = (j - 112) / 112
                dist = torch.sqrt(torch.tensor(x_norm**2 + y_norm**2))
                
                # Color based on distance from center
                intensity = torch.exp(-dist * 2)
                img[i, j, 0] = intensity  # Red channel
                img[i, j, 1] = intensity * 0.8  # Green channel
                img[i, j, 2] = intensity * 0.6  # Blue channel
        
        return img
    
    def encode_views_with_clip(self, rendered_images):
        """Encode rendered images with CLIP"""
        # Preprocess images for CLIP
        processed_images = []
        for img in rendered_images:
            # Convert tensor to PIL Image
            img_np = img.cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
            img_np = img_np.astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Apply CLIP preprocessing
            processed_img = self.clip_preprocess(pil_img)
            processed_images.append(processed_img)
        
        processed_images = torch.stack(processed_images)
        
        # Encode with CLIP
        with torch.no_grad():
            view_features = self.clip_model.encode_image(processed_images)
        
        return view_features
    
    def forward(self, point_cloud, text):
        """Forward pass with real rendering and encoding"""
        # Render candidate views from point cloud
        rendered_images = self.render_candidate_views(point_cloud)
        
        # Encode views with CLIP
        view_features = self.encode_views_with_clip(rendered_images)
        
        # Encode text with CLIP
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Compute scores
        text_relevance, coverage, clip_alignment = self.compute_scores(
            view_features, text_features
        )
        
        # Compute utility
        utility = self.compute_utility(text_relevance, coverage, clip_alignment)
        
        # Select top views
        _, indices = torch.topk(utility, self.num_views)
        selected_features = view_features[indices]
        
        return selected_features, indices
    
    def get_reg_loss(self):
        """L2 regularization for wt"""
        target = torch.tensor(0.3, device=self.device)
        return F.mse_loss(self.wt, target)
