import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import clip
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R
from .point_renderer import create_point_cloud_renderer

class DynamicViewSelection(nn.Module):
    """
    Dynamic View Selection for 3D-R1
    Automatically selects informative 2D views from 3D scenes
    """
    
    def __init__(self, 
                 visual_encoder_name: str = "ViT-B/32",
                 num_candidate_views: int = 8,
                 num_selected_views: int = 4,
                 device: str = "cuda",
                 use_pytorch3d: bool = True):
        super(DynamicViewSelection, self).__init__()
        
        self.num_candidate_views = num_candidate_views
        self.num_selected_views = num_selected_views
        self.device = device
        
        # Load CLIP for visual encoding and alignment
        self.clip_model, self.clip_preprocess = clip.load(visual_encoder_name, device=device)
        
        # Initialize point cloud renderer
        self.renderer = create_point_cloud_renderer(
            use_pytorch3d=use_pytorch3d,
            image_size=224,
            device=device,
            render_method="pulsar" if use_pytorch3d else "fallback"
        )
        
        # Learnable weights for score fusion
        self.wt = nn.Parameter(torch.tensor(0.3))  # text relevance weight
        self.wc = nn.Parameter(torch.tensor(0.35))  # coverage weight
        self.wclip = nn.Parameter(torch.tensor(0.35))  # CLIP alignment weight
        
        # Target value for L2 regularization
        self.target_wt = 0.3
        
        # Visual encoder for feature extraction
        self.visual_encoder = self.clip_model.visual
        
        # Coverage scoring network
        self.coverage_net = nn.Sequential(
            nn.Linear(512, 256),  # CLIP feature dimension
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def generate_candidate_views(self, point_cloud: torch.Tensor, 
                               scene_bounds: torch.Tensor) -> List[Dict]:
        """
        Generate candidate views by sampling camera positions around the scene
        """
        views = []
        
        # Get scene center and size
        scene_center = (scene_bounds[0] + scene_bounds[1]) / 2
        scene_size = scene_bounds[1] - scene_bounds[0]
        max_radius = torch.norm(scene_size) * 1.5
        
        # Generate candidate camera positions
        for i in range(self.num_candidate_views):
            if i == 0:
                # Top-down view
                camera_pos = scene_center + torch.tensor([0, 0, max_radius])
                look_at = scene_center
            elif i == 1:
                # Front view
                camera_pos = scene_center + torch.tensor([0, -max_radius, 0])
                look_at = scene_center
            elif i == 2:
                # Side view
                camera_pos = scene_center + torch.tensor([max_radius, 0, 0])
                look_at = scene_center
            else:
                # Random views around the scene
                angle = 2 * np.pi * i / self.num_candidate_views
                height = np.random.uniform(0.3, 1.0)
                radius = max_radius * np.random.uniform(0.8, 1.2)
                
                camera_pos = scene_center + torch.tensor([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    max_radius * height
                ])
                look_at = scene_center
            
            # Calculate camera orientation
            forward = F.normalize(look_at - camera_pos, dim=-1)
            right = F.normalize(torch.cross(forward, torch.tensor([0, 0, 1.0])), dim=-1)
            up = F.normalize(torch.cross(right, forward), dim=-1)
            
            # Create rotation matrix
            rotation_matrix = torch.stack([right, up, forward], dim=-1)
            
            views.append({
                'position': camera_pos,
                'rotation': rotation_matrix,
                'look_at': look_at
            })
        
        return views
    
    def render_view(self, point_cloud: torch.Tensor, 
                   camera_params: Dict,
                   image_size: int = 224) -> torch.Tensor:
        """
        Render 2D image from 3D point cloud using proper 3D rendering
        """
        return self.renderer.render_point_cloud(point_cloud, camera_params)
    
    def compute_text_relevance_score(self, view_features: torch.Tensor, 
                                   text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute SText→3D score: text relevance to 3D view
        """
        # Normalize features
        view_features = F.normalize(view_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(view_features, text_features.T).squeeze(-1)
        
        return similarity
    
    def compute_coverage_score(self, view_features: torch.Tensor) -> torch.Tensor:
        """
        Compute SImage→3D score: coverage of scene content
        """
        # Use coverage network to predict coverage score
        coverage_scores = self.coverage_net(view_features).squeeze(-1)
        
        # Apply sigmoid to get scores between 0 and 1
        coverage_scores = torch.sigmoid(coverage_scores)
        
        return coverage_scores
    
    def compute_clip_alignment_score(self, view_features: torch.Tensor, 
                                   text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute SCLIP score: CLIP alignment between view and text
        """
        # Normalize features
        view_features = F.normalize(view_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute CLIP-style alignment
        alignment_scores = torch.matmul(view_features, text_features.T).squeeze(-1)
        
        return alignment_scores
    
    def compute_utility_score(self, text_relevance: torch.Tensor,
                            coverage: torch.Tensor,
                            clip_alignment: torch.Tensor) -> torch.Tensor:
        """
        Compute overall utility score U(v) for each view
        """
        # Ensure weights sum to 1 for coverage and CLIP
        wc_normalized = self.wc / (self.wc + self.wclip)
        wclip_normalized = self.wclip / (self.wc + self.wclip)
        
        # Compute utility score
        utility = (self.wt * text_relevance + 
                  wc_normalized * coverage + 
                  wclip_normalized * clip_alignment)
        
        return utility
    
    def select_views(self, utility_scores: torch.Tensor) -> torch.Tensor:
        """
        Select top-k views based on utility scores
        """
        # Select top-k views
        _, indices = torch.topk(utility_scores, self.num_selected_views)
        
        return indices
    
    def forward(self, point_cloud: torch.Tensor, 
               scene_bounds: torch.Tensor,
               text: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass for dynamic view selection
        """
        # Generate candidate views
        candidate_views = self.generate_candidate_views(point_cloud, scene_bounds)
        
        # Render and encode all candidate views
        view_features = []
        
        for view in candidate_views:
            # Render view
            image = self.render_view(point_cloud, view)
            
            # Encode with CLIP
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image.unsqueeze(0))
                view_features.append(image_features.squeeze(0))
        
        view_features = torch.stack(view_features)  # (num_views, feature_dim)
        
        # Encode text with CLIP
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Compute all scores
        text_relevance = self.compute_text_relevance_score(view_features, text_features)
        coverage = self.compute_coverage_score(view_features)
        clip_alignment = self.compute_clip_alignment_score(view_features, text_features)
        
        # Compute utility scores
        utility_scores = self.compute_utility_score(text_relevance, coverage, clip_alignment)
        
        # Select views
        selected_indices = self.select_views(utility_scores)
        selected_view_features = view_features[selected_indices]
        
        # Store scores for analysis
        scores = {
            'text_relevance': text_relevance,
            'coverage': coverage,
            'clip_alignment': clip_alignment,
            'utility': utility_scores,
            'selected_indices': selected_indices
        }
        
        return selected_view_features, selected_indices, scores
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute L2 regularization loss for wt to prevent overly dominant text influence
        """
        return F.mse_loss(self.wt, torch.tensor(self.target_wt, device=self.device))
