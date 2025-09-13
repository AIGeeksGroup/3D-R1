import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

try:
    import pytorch3d
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        PointLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        TexturesVertex,
        look_at_view_transform,
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
        NormWeightedCompositor,
        PulsarRenderer
    )
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.loss import point_mesh_distance
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: PyTorch3D not available. Please install it with: pip install pytorch3d")


class PointCloudRenderer(nn.Module):
    """
    Proper 3D point cloud renderer using PyTorch3D
    """
    
    def __init__(self, 
                 image_size: int = 224,
                 device: str = "cuda",
                 point_size: float = 0.01,
                 render_method: str = "pulsar"):
        super().__init__()
        
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D is required for proper 3D rendering. Please install it.")
        
        self.image_size = image_size
        self.device = device
        self.point_size = point_size
        self.render_method = render_method
        
        # Initialize renderer based on method
        if render_method == "pulsar":
            self.renderer = self._init_pulsar_renderer()
        elif render_method == "points":
            self.renderer = self._init_points_renderer()
        else:
            raise ValueError(f"Unknown render method: {render_method}")
    
    def _init_pulsar_renderer(self):
        """Initialize Pulsar renderer for point clouds"""
        return PulsarRenderer(
            image_size=self.image_size,
            radius_world=self.point_size,
            max_num_balls=100000,
            orthogonal_projection=False
        )
    
    def _init_points_renderer(self):
        """Initialize points renderer"""
        # Rasterization settings
        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=self.point_size,
            points_per_pixel=8
        )
        
        # Create rasterizer
        rasterizer = PointsRasterizer(
            cameras=None,  # Will be set per render
            raster_settings=raster_settings
        )
        
        # Create compositor
        compositor = AlphaCompositor()
        
        # Create renderer
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=compositor
        )
        
        return renderer
    
    def _create_point_cloud(self, points: torch.Tensor, colors: Optional[torch.Tensor] = None) -> Pointclouds:
        """
        Create PyTorch3D Pointclouds object from points and colors
        
        Args:
            points: (N, 3) tensor of 3D points
            colors: (N, 3) tensor of RGB colors, optional
        
        Returns:
            Pointclouds object
        """
        if colors is None:
            # Generate colors based on point positions
            colors = self._generate_colors_from_positions(points)
        
        # Ensure points and colors are on the correct device
        points = points.to(self.device)
        colors = colors.to(self.device)
        
        return Pointclouds(points=[points], features=[colors])
    
    def _generate_colors_from_positions(self, points: torch.Tensor) -> torch.Tensor:
        """
        Generate colors based on point positions
        
        Args:
            points: (N, 3) tensor of 3D points
        
        Returns:
            (N, 3) tensor of RGB colors
        """
        # Normalize points to [0, 1] range
        points_min = points.min(dim=0)[0]
        points_max = points.max(dim=0)[0]
        points_range = points_max - points_min
        points_range = torch.where(points_range == 0, torch.ones_like(points_range), points_range)
        
        normalized_points = (points - points_min) / points_range
        
        # Use position as RGB color
        colors = normalized_points
        
        # Ensure colors are in [0, 1] range
        colors = torch.clamp(colors, 0, 1)
        
        return colors
    
    def _create_camera(self, camera_params: Dict) -> FoVPerspectiveCameras:
        """
        Create PyTorch3D camera from camera parameters
        
        Args:
            camera_params: Dict with 'position', 'rotation', 'look_at' keys
        
        Returns:
            FoVPerspectiveCameras object
        """
        position = camera_params['position'].to(self.device)
        look_at = camera_params['look_at'].to(self.device)
        
        # Calculate camera distance and elevation
        distance = torch.norm(position - look_at)
        
        # Convert to spherical coordinates for FoV camera
        # This is a simplified conversion - in practice you might want more sophisticated camera setup
        R, T = look_at_view_transform(
            dist=distance.item(),
            elev=0,  # You might want to calculate this from position
            azim=0,  # You might want to calculate this from position
            device=self.device
        )
        
        # Create camera
        camera = FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=60,  # Field of view in degrees
            device=self.device
        )
        
        return camera
    
    def render_point_cloud(self, 
                          point_cloud: torch.Tensor,
                          camera_params: Dict,
                          colors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Render point cloud from given camera viewpoint
        
        Args:
            point_cloud: (N, 3) tensor of 3D points
            camera_params: Dict with camera parameters
            colors: (N, 3) tensor of RGB colors, optional
        
        Returns:
            (3, H, W) tensor of rendered image
        """
        # Create point cloud object
        pcl = self._create_point_cloud(point_cloud, colors)
        
        # Create camera
        camera = self._create_camera(camera_params)
        
        if self.render_method == "pulsar":
            return self._render_with_pulsar(pcl, camera)
        elif self.render_method == "points":
            return self._render_with_points(pcl, camera)
        else:
            raise ValueError(f"Unknown render method: {self.render_method}")
    
    def _render_with_pulsar(self, pcl: Pointclouds, camera: FoVPerspectiveCameras) -> torch.Tensor:
        """Render using Pulsar renderer"""
        # Pulsar renderer expects specific input format
        points = pcl.points_list()[0]  # (N, 3)
        colors = pcl.features_list()[0]  # (N, 3)
        
        # Convert colors to [0, 1] range if needed
        if colors.max() > 1.0:
            colors = colors / 255.0
        
        # Render with Pulsar
        rendered_image = self.renderer(
            points.unsqueeze(0),  # Add batch dimension
            colors.unsqueeze(0),  # Add batch dimension
            camera.R,
            camera.T
        )
        
        # Remove batch dimension and convert to (3, H, W)
        rendered_image = rendered_image.squeeze(0).permute(2, 0, 1)
        
        return rendered_image
    
    def _render_with_points(self, pcl: Pointclouds, camera: FoVPerspectiveCameras) -> torch.Tensor:
        """Render using points renderer"""
        # Update camera in rasterizer
        self.renderer.rasterizer.cameras = camera
        
        # Render
        rendered_image = self.renderer(pcl)
        
        # Convert to (3, H, W) format
        rendered_image = rendered_image.squeeze(0).permute(2, 0, 1)
        
        return rendered_image
    
    def render_multiple_views(self, 
                             point_cloud: torch.Tensor,
                             camera_params_list: List[Dict],
                             colors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Render multiple views of the same point cloud
        
        Args:
            point_cloud: (N, 3) tensor of 3D points
            camera_params_list: List of camera parameter dicts
            colors: (N, 3) tensor of RGB colors, optional
        
        Returns:
            (num_views, 3, H, W) tensor of rendered images
        """
        rendered_images = []
        
        for camera_params in camera_params_list:
            rendered_image = self.render_point_cloud(point_cloud, camera_params, colors)
            rendered_images.append(rendered_image)
        
        return torch.stack(rendered_images)
    
    def forward(self, 
                point_cloud: torch.Tensor,
                camera_params: Dict,
                colors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for rendering
        
        Args:
            point_cloud: (N, 3) tensor of 3D points
            camera_params: Dict with camera parameters
            colors: (N, 3) tensor of RGB colors, optional
        
        Returns:
            (3, H, W) tensor of rendered image
        """
        return self.render_point_cloud(point_cloud, camera_params, colors)


class FallbackRenderer(nn.Module):
    """
    Fallback renderer when PyTorch3D is not available
    Uses improved point splatting instead of the simple placeholder
    """
    
    def __init__(self, image_size: int = 224, device: str = "cuda"):
        super().__init__()
        self.image_size = image_size
        self.device = device
    
    def render_point_cloud(self, 
                          point_cloud: torch.Tensor,
                          camera_params: Dict,
                          colors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Improved point cloud rendering without PyTorch3D
        """
        # Transform points to camera coordinates
        camera_pos = camera_params['position'].to(self.device)
        rotation_matrix = camera_params['rotation'].to(self.device)
        
        # Transform points
        points_cam = torch.matmul(point_cloud - camera_pos, rotation_matrix.T)
        
        # Filter points in front of camera
        valid_points = points_cam[:, 2] > 0.1  # Points behind camera
        if not valid_points.any():
            return torch.zeros(3, self.image_size, self.image_size, device=self.device)
        
        points_cam = points_cam[valid_points]
        
        # Project to 2D with proper perspective projection
        focal_length = 500.0
        points_2d = points_cam[:, :2] * focal_length / points_cam[:, 2:3]
        
        # Convert to image coordinates
        points_2d[:, 0] = points_2d[:, 0] + self.image_size // 2  # x coordinate
        points_2d[:, 1] = points_2d[:, 1] + self.image_size // 2  # y coordinate
        
        # Create image
        image = torch.zeros(3, self.image_size, self.image_size, device=self.device)
        
        # Generate colors if not provided
        if colors is None:
            colors = self._generate_colors_from_positions(point_cloud[valid_points])
        else:
            colors = colors[valid_points]
        
        # Point splatting with depth sorting
        depths = points_cam[:, 2]
        sorted_indices = torch.argsort(depths, descending=True)  # Render far points first
        
        for idx in sorted_indices:
            x, y = points_2d[idx, 0], points_2d[idx, 1]
            
            # Check if point is within image bounds
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                # Simple point splatting (you could improve this with proper splatting)
                x_int, y_int = int(x), int(y)
                if 0 <= x_int < self.image_size and 0 <= y_int < self.image_size:
                    image[:, y_int, x_int] = colors[idx]
        
        return image
    
    def _generate_colors_from_positions(self, points: torch.Tensor) -> torch.Tensor:
        """Generate colors based on point positions"""
        # Normalize points to [0, 1] range
        points_min = points.min(dim=0)[0]
        points_max = points.max(dim=0)[0]
        points_range = points_max - points_min
        points_range = torch.where(points_range == 0, torch.ones_like(points_range), points_range)
        
        normalized_points = (points - points_min) / points_range
        colors = torch.clamp(normalized_points, 0, 1)
        
        return colors


def create_point_cloud_renderer(use_pytorch3d: bool = True, 
                               image_size: int = 224,
                               device: str = "cuda",
                               **kwargs) -> nn.Module:
    """
    Factory function to create appropriate point cloud renderer
    
    Args:
        use_pytorch3d: Whether to use PyTorch3D renderer
        image_size: Size of rendered images
        device: Device to use
        **kwargs: Additional arguments for renderer
    
    Returns:
        Point cloud renderer
    """
    if use_pytorch3d and PYTORCH3D_AVAILABLE:
        return PyTorch3DPointCloudRenderer(
            image_size=image_size,
            device=device,
            **kwargs
        )
    else:
        print("Using fallback renderer (PyTorch3D not available)")
        return FallbackRenderer(
            image_size=image_size,
            device=device
        )
