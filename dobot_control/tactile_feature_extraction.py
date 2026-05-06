"""
Tactile-Visual Feature Extraction Module

This module implements the extraction of multi-modal features from tactile sensors
and camera images for the HSA (Hard Sample Aware) contrastive loss.

Key components:
1. Forward kinematics to compute 3D pose of tactile sensor
2. Camera projection to find 2D bounding boxes (only for external/stationary cameras)
3. CLIP-like vision backbone to extract intermediate features
4. Feature extraction: h_tau (tactile), h_w (wrist), h_tp (third-person)

Important Design Note:
- **Wrist camera**: Mounted on robot with tactile sensor → Fixed relative position
  → NO bounding box computation needed → Simply use all features
  
- **Third-person/top camera**: Stationary in workspace → Sensor moves relative to camera
  → Bounding box computation REQUIRED using forward kinematics + camera projection
  → Select features within bounding box

This design avoids unnecessary computation for the common case where wrist camera
and tactile sensor are rigidly mounted together.
"""

import numpy as np
from typing import Tuple, Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. TactileFeatureExtractor will not work.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Some image operations may fail.")


class CLIPLikeBackbone(nn.Module if TORCH_AVAILABLE else object):
    """
    A CLIP-like vision backbone for extracting intermediate features.
    
    DEPRECATED: This custom implementation is being replaced by actual CLIP from open_clip.
    Kept for backward compatibility only.
    """
    
    def __init__(self, 
                 model_name: str = "vit_b_16",
                 img_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 return_intermediate: bool = True):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CLIPLikeBackbone")
        super().__init__()
        print("WARNING: CLIPLikeBackbone is deprecated. Use open_clip CLIP encoder instead.")
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patches_per_side = img_size // patch_size
        self.return_intermediate = return_intermediate
        
        # Patch embedding
        self.conv_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.ln_post = nn.LayerNorm(embed_dim)
        
        # Normalize input
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, x: torch.Tensor, return_all_layers: bool = False) -> torch.Tensor:
        """
        Forward pass through the vision backbone.
        
        Args:
            x: Input images, shape (B, C, H, W) where H=W=img_size
            return_all_layers: If True, return features from all transformer layers
        
        Returns:
            If return_all_layers: List of feature tensors from each layer
            Otherwise: Features from intermediate layer (shape: B, N_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Normalize
        x = self.normalize(x)
        
        # Patch embedding
        x = self.conv_proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, E, HP, WP = x.shape
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        
        # Add class token
        class_token = self.class_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([class_token, x], dim=1)  # (B, N_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.positional_embedding
        
        # Store intermediate features
        intermediate_features = []
        
        # Pass through transformer
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            if self.return_intermediate and i == self.num_layers // 2:
                # Store intermediate layer (skip class token)
                intermediate_features.append(x[:, 1:, :])  # (B, N_patches, embed_dim)
        
        if return_all_layers:
            return intermediate_features if intermediate_features else [x[:, 1:, :]]
        else:
            # Return intermediate layer features or final features
            if intermediate_features:
                return intermediate_features[-1]  # (B, N_patches, embed_dim)
            else:
                x = self.ln_post(x)
                return x[:, 1:, :]  # (B, N_patches, embed_dim)
    
    def get_feature_grid_shape(self) -> Tuple[int, int]:
        """Get the spatial shape of the feature grid (height, width)."""
        return (self.patches_per_side, self.patches_per_side)


class ForwardKinematics:
    """Forward kinematics calculator for DoBot robot."""
    
    @staticmethod
    def dh_transformation_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """Create a DH transformation matrix."""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        return np.array([
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def claw_width(coef: float) -> float:
        """
        Calculate the claw width
        """
        claw_servo = 2.3818 - coef * 1.5401
        cos_claw_servo = np.cos(claw_servo)
        claw_wid = 0.03 * cos_claw_servo + 0.5 * np.sqrt(0.0036 * cos_claw_servo ** 2 + 0.0028)
        return claw_wid

    def forward_kinematics(self, q0, q1, q2, q3, q4, q5, y, r_type):
        """
        Compute the forward kinematics
        """
        if r_type == "Nova 2":
            dh_params = [
                (q0, 0.2234, 0, np.pi / 2),
                (q1 - np.pi / 2, 0, -0.280, 0),
                (q2, 0, -0.225, 0),
                (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
                (q4, 0.120, 0, -np.pi / 2),
                (q5, 0.088, 0, 0)
            ]
        if r_type == "Nova 5":
            dh_params = [
                (q0, 0.240, 0, np.pi / 2),
                (q1 - np.pi / 2, 0, -0.400, 0),
                (q2, 0, -0.330, 0),
                (q3 - np.pi / 2, 0.135, 0, np.pi / 2),
                (q4, 0.120, 0, -np.pi / 2),
                (q5, 0.088, 0, 0)
            ]

        t = np.eye(4)
        for params in dh_params:
            t = np.dot(t, self.dh_transformation_matrix(*params))
        t_tool = np.eye(4)
        t_tool[:3, 3] = np.array([0, y, 0.2])
        t_final = np.dot(t, t_tool)
        pos = t_final[:3, 3]
        return pos
    
    @staticmethod
    def compute_tactile_sensor_pose(joint_angles: np.ndarray, 
                                    robot_type: str = "Nova 2",
                                    sensor_offset: np.ndarray = np.array([0, 0, 0.02])) -> np.ndarray:
        """
        Compute the 3D pose (4x4 transformation matrix) of the tactile sensor.
        
        Args:
            joint_angles: Joint angles in radians, shape (6,)
            robot_type: Robot type ("Nova 2" or "Nova 5")
            sensor_offset: Offset from end-effector to sensor center, shape (3,)
        
        Returns:
            4x4 transformation matrix representing the sensor pose in SE(3)
        """
        if len(joint_angles) < 6:
            raise ValueError("Joint angles must have at least 6 values")
        
        q0, q1, q2, q3, q4, q5 = joint_angles[:6]
        
        if robot_type == "Nova 2":
            dh_params = [
                (q0, 0.2234, 0, np.pi / 2),
                (q1 - np.pi / 2, 0, -0.280, 0),
                (q2, 0, -0.225, 0),
                (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
                (q4, 0.120, 0, -np.pi / 2),
                (q5, 0.088, 0, 0)
            ]
        elif robot_type == "Nova 5":
            dh_params = [
                (q0, 0.240, 0, np.pi / 2),
                (q1 - np.pi / 2, 0, -0.400, 0),
                (q2, 0, -0.330, 0),
                (q3 - np.pi / 2, 0.135, 0, np.pi / 2),
                (q4, 0.120, 0, -np.pi / 2),
                (q5, 0.088, 0, 0)
            ]
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
        
        # Compute forward kinematics
        T = np.eye(4)
        for params in dh_params:
            T = np.dot(T, ForwardKinematics.dh_transformation_matrix(*params))
        
        # Apply sensor offset
        T_sensor_offset = np.eye(4)
        T_sensor_offset[:3, 3] = sensor_offset
        T_sensor = np.dot(T, T_sensor_offset)
        
        return T_sensor


class CameraProjection:
    """Camera projection utilities for 3D to 2D mapping."""
    
    @staticmethod
    def project_3d_to_2d(points_3d: np.ndarray,
                         K: np.ndarray,
                         E: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: 3D points in world coordinates, shape (N, 3) or (3,)
            K: Camera intrinsic matrix, shape (3, 3)
            E: Camera extrinsic matrix (world to camera), shape (4, 4)
        
        Returns:
            2D pixel coordinates, shape (N, 2) or (2,)
        """
        points_3d = np.array(points_3d)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)
        
        # Add homogeneous coordinate
        N = points_3d.shape[0]
        points_3d_homo = np.hstack([points_3d, np.ones((N, 1))])
        
        # Transform to camera coordinates
        points_cam = (E @ points_3d_homo.T).T  # (N, 4)
        
        # Project to 2D
        points_cam_3d = points_cam[:, :3]  # (N, 3)
        points_2d_homo = (K @ points_cam_3d.T).T  # (N, 3)
        
        # Divide by z to get pixel coordinates
        z = points_2d_homo[:, 2:3]
        z = np.where(z == 0, 1e-8, z)  # Avoid division by zero
        points_2d = points_2d_homo[:, :2] / z
        
        if points_2d.shape[0] == 1:
            return points_2d[0]
        return points_2d
    
    @staticmethod
    def compute_sensor_bounding_box(sensor_pose: np.ndarray,
                                     sensor_size: Tuple[float, float],
                                     K: np.ndarray,
                                     E: np.ndarray,
                                     img_size: Tuple[int, int]) -> Dict[str, float]:
        """
        Compute 2D bounding box of tactile sensor in image.
        
        Args:
            sensor_pose: 4x4 transformation matrix of sensor
            sensor_size: (width, height) of sensor in meters
            K: Camera intrinsic matrix
            E: Camera extrinsic matrix
            img_size: (width, height) of image
        
        Returns:
            Dictionary with keys: x_min, y_min, x_max, y_max
        """
        # Define sensor corners in sensor frame
        w, h = sensor_size
        corners_sensor = np.array([
            [-w/2, -h/2, 0],
            [w/2, -h/2, 0],
            [w/2, h/2, 0],
            [-w/2, h/2, 0],
        ])
        
        # Transform to world coordinates
        corners_sensor_homo = np.hstack([corners_sensor, np.ones((4, 1))])
        corners_world = (sensor_pose @ corners_sensor_homo.T).T[:, :3]
        
        # Project to 2D
        corners_2d = CameraProjection.project_3d_to_2d(corners_world, K, E)
        
        # Compute bounding box
        x_min = max(0, int(np.floor(np.min(corners_2d[:, 0]))))
        y_min = max(0, int(np.floor(np.min(corners_2d[:, 1]))))
        x_max = min(img_size[0] - 1, int(np.ceil(np.max(corners_2d[:, 0]))))
        y_max = min(img_size[1] - 1, int(np.ceil(np.max(corners_2d[:, 1]))))
        
        return {
            'x_min': float(x_min),
            'y_min': float(y_min),
            'x_max': float(x_max),
            'y_max': float(y_max)
        }


class TactileFeatureExtractor:
    """Main class for extracting tactile and visual features using external CLIP encoder."""
    
    def __init__(self,
                 clip_encoder=None,
                 img_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 device: str = 'cpu'):
        """
        Initialize the feature extractor.
        
        Args:
            clip_encoder: External CLIP encoder from policy (preferred). If None, creates legacy CLIPLikeBackbone.
            img_size: Input image size (will be resized to this)
            patch_size: Patch size for Vision Transformer
            embed_dim: Embedding dimension
            num_heads: Number of attention heads (must divide embed_dim evenly)
            device: Device to run on ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TactileFeatureExtractor")
        self.device = torch.device(device)
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Use external CLIP encoder if provided
        if clip_encoder is not None:
            print("TactileFeatureExtractor: Using external CLIP encoder from policy")
            self.backbone = clip_encoder
            self.use_external_clip = True
            # Get feature grid shape from CLIP encoder
            if hasattr(clip_encoder, 'num_patches_per_side'):
                self.feature_grid_shape = (clip_encoder.num_patches_per_side, clip_encoder.num_patches_per_side)
            else:
                # Default for CLIP ViT-B/16
                self.feature_grid_shape = (img_size // patch_size, img_size // patch_size)
        else:
            # Fallback to legacy custom backbone (deprecated)
            print("TactileFeatureExtractor: Using legacy CLIPLikeBackbone (deprecated)")
            self.backbone = CLIPLikeBackbone(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads
            ).to(self.device)
            self.use_external_clip = False
            self.feature_grid_shape = self.backbone.get_feature_grid_shape()
        
        # Note: backbone starts in training mode by default
        # Policy will control train/eval mode during training/inference
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # Gripper-aware offset configuration (calibrated from real robot)
        # NOTE: Requires real camera intrinsic matrix (K) to be provided during feature extraction
        # No fake/default camera parameters will be used as fallback
        self.use_gripper_offset = True
        self.wrist_camera_position = np.array([0.0, 0.05, 0.065])  # meters, in gripper frame
        self.tactile_sensor_size = (0.03, 0.04)  # meters, physical sensor dimensions
    
    def map_bbox_to_feature_grid(self,
                                  bbox: Dict[str, float],
                                  original_img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Map bounding box coordinates from original image to feature grid.
        
        Args:
            bbox: Bounding box dict with x_min, y_min, x_max, y_max
            original_img_size: (width, height) of original image
        
        Returns:
            (feat_x_min, feat_y_min, feat_x_max, feat_y_max) as integer indices
        """
        orig_w, orig_h = original_img_size
        feat_h, feat_w = self.feature_grid_shape
        
        # Calculate scaling factors
        scale_w = orig_w / feat_w
        scale_h = orig_h / feat_h
        
        # Map coordinates
        feat_x_min = int(np.floor(bbox['x_min'] / scale_w))
        feat_y_min = int(np.floor(bbox['y_min'] / scale_h))
        feat_x_max = int(np.floor(bbox['x_max'] / scale_w))
        feat_y_max = int(np.floor(bbox['y_max'] / scale_h))

        # Ensure ordering (swap if needed)
        if feat_x_min > feat_x_max:
            feat_x_min, feat_x_max = feat_x_max, feat_x_min
        if feat_y_min > feat_y_max:
            feat_y_min, feat_y_max = feat_y_max, feat_y_min
        
        # Clamp to valid range
        feat_x_min = max(0, min(feat_x_min, feat_w - 1))
        feat_y_min = max(0, min(feat_y_min, feat_h - 1))
        feat_x_max = max(0, min(feat_x_max, feat_w - 1))
        feat_y_max = max(0, min(feat_y_max, feat_h - 1))
        
        return feat_x_min, feat_y_min, feat_x_max, feat_y_max
    
    def extract_wrist_features(self,
                                wrist_image: np.ndarray,
                                gripper_width: Optional[float] = None,
                                camera_K: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Extract h_w: mean-pooled features from wrist-view tokens.
        
        With gripper-aware offset: computes sensor position based on gripper width,
        projects to 2D, and extracts features from bounding box region.
        
        Args:
            wrist_image: Wrist camera image, shape (H, W, 3)
            gripper_width: Gripper clamp width in meters (ONE side position from center)
            camera_K: Camera intrinsic matrix (3, 3) - REQUIRED for gripper-aware offset
                     Must be real calibrated camera parameters, no fake fallback provided
        
        Returns:
            h_w: Mean-pooled feature vector, shape (embed_dim,)
        
        Note:
            If camera_K is not provided, falls back to mean pooling all features (no gripper offset)
        """
        # Preprocess image
        img_tensor = self.transform(wrist_image).unsqueeze(0).to(self.device)
        
        # Extract features (gradients enabled for training)
        if self.use_external_clip:
            # External CLIP encoder returns (features, pos)
            # features: (1, hidden_dim, N_patches), need to transpose
            features, _ = self.backbone(img_tensor)
            features = features.permute(0, 2, 1)  # (1, N_patches, hidden_dim)
        else:
            # Legacy backbone returns features directly
            features = self.backbone(img_tensor)  # (1, N_patches, embed_dim)
        
        # Compute gripper-based offset if enabled
        if self.use_gripper_offset and gripper_width is not None and camera_K is not None:
            # Using real camera calibration data
            # Tactile sensor 3D position (clamp_width is ONE side, not total)
            # Left finger with sensor moves positively along x when opening
            tactile_3d_pos = np.array([gripper_width, 0.0, 0.0])
            
            # Sensor position relative to camera
            relative_pos = tactile_3d_pos - self.wrist_camera_position
            
            # Project to 2D using pinhole camera model
            fx, fy = camera_K[0, 0], camera_K[1, 1]
            cx, cy = camera_K[0, 2], camera_K[1, 2]
            
            if relative_pos[2] > 1e-6:  # Sensor in front of camera
                # Compute center pixel
                x_center = fx * (relative_pos[0] / relative_pos[2]) + cx
                y_center = fy * (relative_pos[1] / relative_pos[2]) + cy
                
                # Estimate bbox size in pixels (rough approximation based on sensor size and depth)
                # This assumes sensor is roughly perpendicular to camera
                pixel_scale = fx / relative_pos[2]  # pixels per meter at this depth
                bbox_w_pixels = self.tactile_sensor_size[0] * pixel_scale
                bbox_h_pixels = self.tactile_sensor_size[1] * pixel_scale
                
                # Create bounding box
                bbox = {
                    'x_min': x_center - bbox_w_pixels / 2,
                    'y_min': y_center - bbox_h_pixels / 2,
                    'x_max': x_center + bbox_w_pixels / 2,
                    'y_max': y_center + bbox_h_pixels / 2
                }
                
                # Scale bbox to resized image
                orig_h, orig_w = wrist_image.shape[:2]
                scale_to_square = min(self.img_size / orig_w, self.img_size / orig_h)
                bbox_scaled = {
                    'x_min': bbox['x_min'] * scale_to_square,
                    'y_min': bbox['y_min'] * scale_to_square,
                    'x_max': bbox['x_max'] * scale_to_square,
                    'y_max': bbox['y_max'] * scale_to_square
                }
                
                # Map to feature grid
                feat_x_min, feat_y_min, feat_x_max, feat_y_max = self.map_bbox_to_feature_grid(
                    bbox_scaled, (self.img_size, self.img_size)
                )
                
                # Reshape and select features (same logic as extract_third_person_features)
                feat_h, feat_w = self.feature_grid_shape
                features_spatial = features.view(1, feat_h, feat_w, -1)
                selected_features = features_spatial[:, feat_y_min:feat_y_max+1, feat_x_min:feat_x_max+1, :]
                
                # Handle empty selection
                if selected_features.numel() == 0:
                    safe_y = max(0, min((feat_y_min + feat_y_max) // 2, feat_h - 1))
                    safe_x = max(0, min((feat_x_min + feat_x_max) // 2, feat_w - 1))
                    selected_features = features_spatial[:, safe_y:safe_y+1, safe_x:safe_x+1, :]
                
                # Mean pool selected features
                selected_features = selected_features.reshape(-1, features.shape[-1])
                h_w = torch.nanmean(selected_features, dim=0)
                h_w = torch.nan_to_num(h_w, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Fallback: invalid geometry
                h_w = features.mean(dim=1).squeeze(0)
        else:
            # Fallback: no gripper info, use all features
            h_w = features.mean(dim=1).squeeze(0)
        
        return h_w
    
    def extract_third_person_features(self,
                                       tp_image: np.ndarray,
                                       bbox: Dict[str, float]) -> torch.Tensor:
        """
        Extract h_tp: mean-pooled features from third-person camera within bounding box.
        
        For third-person/top camera (stationary in workspace), the tactile sensor
        moves with the robot, so we need to dynamically compute where it appears
        using forward kinematics and camera projection.
        
        Args:
            tp_image: Third-person camera image, shape (H, W, 3)
            bbox: Bounding box dict with x_min, y_min, x_max, y_max
        
        Returns:
            h_tp: Mean-pooled feature vector, shape (embed_dim,)
        """
        # Preprocess image
        img_tensor = self.transform(tp_image).unsqueeze(0).to(self.device)
        
        # Extract features (gradients enabled for training)
        if self.use_external_clip:
            # External CLIP encoder returns (features, pos)
            features, _ = self.backbone(img_tensor)
            features = features.permute(0, 2, 1)  # (1, N_patches, hidden_dim)
        else:
            # Legacy backbone returns features directly
            features = self.backbone(img_tensor)  # (1, N_patches, embed_dim)
        
        # Map bounding box to feature grid
        orig_h, orig_w = tp_image.shape[:2]
        
        # Scale bbox coordinates to match resized image (square, img_size x img_size)
        scale_to_square = min(self.img_size / orig_w, self.img_size / orig_h)
        bbox_scaled = {
            'x_min': bbox['x_min'] * scale_to_square,
            'y_min': bbox['y_min'] * scale_to_square,
            'x_max': bbox['x_max'] * scale_to_square,
            'y_max': bbox['y_max'] * scale_to_square
        }
        
        feat_x_min, feat_y_min, feat_x_max, feat_y_max = self.map_bbox_to_feature_grid(
            bbox_scaled, (self.img_size, self.img_size)
        )
        
        # Reshape features to spatial grid: (1, H_feat, W_feat, embed_dim)
        feat_h, feat_w = self.feature_grid_shape
        features_spatial = features.view(1, feat_h, feat_w, -1)
        
        # Select tokens within bounding box
        selected_features = features_spatial[:, 
                                            feat_y_min:feat_y_max+1, 
                                            feat_x_min:feat_x_max+1, 
                                            :]  # (1, H_box, W_box, embed_dim)

        # Handle empty selections robustly
        if selected_features.numel() == 0:
            # Fallback: pick nearest valid token index within grid
            feat_h, feat_w = self.feature_grid_shape
            safe_y = max(0, min((feat_y_min + feat_y_max) // 2, feat_h - 1))
            safe_x = max(0, min((feat_x_min + feat_x_max) // 2, feat_w - 1))
            selected_features = features_spatial[:, safe_y:safe_y+1, safe_x:safe_x+1, :]

        # Flatten and mean pool (nan-safe)
        selected_features = selected_features.reshape(-1, features.shape[-1])  # (N_tokens, embed_dim)
        h_tp = torch.nanmean(selected_features, dim=0)  # (embed_dim,)
        h_tp = torch.nan_to_num(h_tp, nan=0.0, posinf=0.0, neginf=0.0)
        
        return h_tp
    
    def extract_tactile_features(self,
                                  tactile_image: np.ndarray) -> torch.Tensor:
        """
        Extract h_tau: mean-pooled features from tactile tokens.
        
        Args:
            tactile_image: Tactile sensor image, shape (H, W, 3)
        
        Returns:
            h_tau: Mean-pooled feature vector, shape (embed_dim,)
        """
        # Preprocess image
        img_tensor = self.transform(tactile_image).unsqueeze(0).to(self.device)
        
        # Extract features (gradients enabled for training)
        if self.use_external_clip:
            # External CLIP encoder returns (features, pos)
            features, _ = self.backbone(img_tensor)
            features = features.permute(0, 2, 1)  # (1, N_patches, hidden_dim)
        else:
            # Legacy backbone returns features directly
            features = self.backbone(img_tensor)  # (1, N_patches, embed_dim)
        
        # Mean pool over all tokens
        h_tau = features.mean(dim=1).squeeze(0)  # (embed_dim,)
        
        return h_tau
    
    def extract_features(self,
                         wrist_image: np.ndarray,
                         tactile_image: np.ndarray,
                         gripper_width: Optional[float] = None,
                         camera_K_wrist: Optional[np.ndarray] = None,
                         tp_image: Optional[np.ndarray] = None,
                         bbox_tp: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract all features: h_tau, h_w, and optionally h_tp.
        
        Args:
            wrist_image: Wrist camera image (mounted on robot with tactile sensor)
            tactile_image: Tactile sensor image
            gripper_width: Gripper clamp width in meters (for gripper-aware wrist features)
            camera_K_wrist: Wrist camera intrinsic matrix (3, 3), optional
            tp_image: Optional third-person camera image (stationary in workspace)
            bbox_tp: Optional bounding box in third-person view (required if tp_image provided)
        
        Returns:
            Dictionary with keys: 'h_tau', 'h_w', and optionally 'h_tp'
        
        Note:
            - Wrist camera uses gripper-aware offset (computes bbox from gripper width)
            - Third-person camera DOES need bounding box (stationary camera, moving sensor)
        """
        h_tau = self.extract_tactile_features(tactile_image)
        h_w = self.extract_wrist_features(wrist_image, gripper_width=gripper_width, camera_K=camera_K_wrist)
        
        result = {
            'h_tau': h_tau,
            'h_w': h_w
        }
        
        # Third-person camera features (requires bounding box computation)
        if tp_image is not None and bbox_tp is not None:
            h_tp = self.extract_third_person_features(tp_image, bbox_tp)
            result['h_tp'] = h_tp
        
        return result


# def generate_fake_camera_params(img_size: Tuple[int, int] = (640, 480)) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Generate fake camera intrinsic and extrinsic parameters.
    
#     Args:
#         img_size: (width, height) of image
    
#     Returns:
#         K: Intrinsic matrix (3, 3)
#         E: Extrinsic matrix (4, 4) - world to camera transformation
#     """
#     w, h = img_size
#     fx = fy = 500.0  # Focal length in pixels
#     cx = w / 2.0
#     cy = h / 2.0
    
#     K = np.array([
#         [fx, 0, cx],
#         [0, fy, cy],
#         [0, 0, 1]
#     ])
    
#     # Fake extrinsic: camera positioned at (0.5, 0.5, 0.5) looking at origin
#     position = np.array([0.5, 0.5, 0.5])
#     target = np.array([0, 0, 0])
#     up = np.array([0, 0, 1])
    
#     # Compute rotation
#     z_axis = target - position
#     z_axis = z_axis / np.linalg.norm(z_axis)
#     x_axis = np.cross(up, z_axis)
#     x_axis = x_axis / np.linalg.norm(x_axis)
#     y_axis = np.cross(z_axis, x_axis)
    
#     R = np.array([x_axis, y_axis, z_axis])
#     t = -R @ position
    
#     E = np.eye(4)
#     E[:3, :3] = R
#     E[:3, 3] = t
    
#     return K, E


# def generate_fake_images(img_size: Tuple[int, int] = (640, 480)) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Generate fake wrist and tactile images.
    
#     Args:
#         img_size: (width, height) of images
    
#     Returns:
#         wrist_image: Fake wrist camera image (H, W, 3)
#         tactile_image: Fake tactile sensor image (H, W, 3)
#     """
#     w, h = img_size
    
#     # Generate wrist image: random pattern with some structure
#     wrist_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
#     # Add some colored patches to make it more realistic
#     if CV2_AVAILABLE:
#         cv2.rectangle(wrist_image, (100, 150), (300, 350), (255, 0, 0), -1)
#         cv2.circle(wrist_image, (400, 200), 50, (0, 255, 0), -1)
#     else:
#         # Manual drawing without cv2
#         wrist_image[150:350, 100:300] = [255, 0, 0]
    
#     # Generate tactile image: smaller, more uniform pattern
#     tactile_h, tactile_w = 100, 100
#     tactile_image = np.random.randint(0, 255, (tactile_h, tactile_w, 3), dtype=np.uint8)
    
#     # Add circular pattern to simulate contact
#     if CV2_AVAILABLE:
#         cv2.circle(tactile_image, (50, 50), 30, (255, 255, 255), -1)
#     else:
#         # Manual drawing
#         y, x = np.ogrid[:tactile_h, :tactile_w]
#         mask = (x - 50)**2 + (y - 50)**2 <= 30**2
#         tactile_image[mask] = [255, 255, 255]
    
#     return wrist_image, tactile_image

