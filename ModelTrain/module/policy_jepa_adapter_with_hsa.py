"""
Enhanced ACTJEPAAdapter Policy with HSA (Hard Sample Aware) Loss

This module extends the ACTJEPAAdapter policy to include tactile-visual feature alignment
using HSA contrastive loss.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

# Add parent directories to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'dobot_control'))

from dobot_control.tactile_feature_extraction import (
    TactileFeatureExtractor,
    ForwardKinematics,
    CameraProjection,
)
from dobot_control.hsa_loss import HSALossWithThirdPerson

# Import the base ACTJEPAAdapterPolicy
from ModelTrain.module.policy_jepa_adapter import ACTJEPAAdapterPolicy


class ACTJEPAHsa(ACTJEPAAdapterPolicy):
    """
    Enhanced ACTJEPAAdapter Policy with tactile-visual feature alignment via HSA loss.
    
    This extends the ACTJEPAAdapter policy to:
    1. Extract tactile and wrist visual features
    2. Compute HSA contrastive loss for feature alignment
    3. Combine ACTJEPAAdapter loss (L1 + KL) with HSA loss
    """
    
    def __init__(self, args_override, hsa_config: Optional[Dict] = None):
        """
        Initialize enhanced ACTJEPAAdapter policy with HSA loss.
        
        Args:
            args_override: Configuration for ACTJEPAAdapter model
            hsa_config: Configuration for HSA loss, with keys:
                - enable_hsa (bool): Whether to enable HSA loss
                - hsa_weight (float): Weight for HSA loss term
                - temperature (float): Temperature for contrastive loss
                - use_third_person (bool): Whether to use third-person camera
                - feature_dim (int): Feature dimension for CLIP backbone
                - img_size (int): Image size for feature extractor
                - patch_size (int): Patch size for ViT
                - camera_params (dict): Camera intrinsic/extrinsic parameters
                - robot_type (str): Robot type for FK ("Nova 2" or "Nova 5")
                - sensor_offset (np.ndarray): Sensor offset from end-effector
                - wrist_camera_idx (int): Index of wrist camera in RGB camera_names
                - tactile_camera_idx (int): Index of tactile sensor in tactile_camera_names
        """
        super().__init__(args_override)
        
        # Parse HSA configuration
        if hsa_config is None:
            hsa_config = {}
        
        self.enable_hsa = hsa_config.get('enable_hsa', False)
        self.hsa_weight = hsa_config.get('hsa_weight', 1.0)
        
        if self.enable_hsa:
            # Initialize tactile feature extractor with CLIP encoder from policy
            feature_dim = hsa_config.get('feature_dim', 768)
            img_size = hsa_config.get('img_size', 224)
            patch_size = hsa_config.get('patch_size', 16)
            num_heads = hsa_config.get('num_heads', 12)  # Default 12 for ViT-L (768), use 16 for ViT-G (1408)
            
            # Get CLIP encoder from policy model if available
            clip_encoder = getattr(self.model, 'clip_encoder', None) if hasattr(self, 'model') else None
            
            self.feature_extractor = TactileFeatureExtractor(
                clip_encoder=clip_encoder,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=feature_dim,
                num_heads=num_heads,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            if clip_encoder is not None:
                print("HSA: Using shared CLIP encoder from policy")
            else:
                print("HSA: CLIP encoder not found, using legacy custom backbone")
            
            # Initialize HSA loss
            temperature = hsa_config.get('temperature', 0.07)
            use_third_person = hsa_config.get('use_third_person', False)
            tp_weight = hsa_config.get('tp_weight', 0.5)
            
            self.hsa_loss_fn = HSALossWithThirdPerson(
                temperature=temperature,
                use_third_person=use_third_person,
                tp_weight=tp_weight,
                reduction='mean'
            )
            
            # Camera and robot configuration
            self.camera_params = hsa_config.get('camera_params', None)
            self.robot_type = hsa_config.get('robot_type', 'Nova 2')
            self.sensor_offset = hsa_config.get('sensor_offset', np.array([0.0, 0.0, 0.02]))
            self.sensor_size = hsa_config.get('sensor_size', (0.04, 0.04))  # 4cm x 4cm
            
            # Camera indices
            # RGB cameras order: ['top', 'left_wrist', 'right_wrist']
            # wrist_camera_idx is the index within RGB cameras (first element of image list)
            # tactile_camera_idx is the index within tactile cameras (second element of image list)
            self.wrist_camera_idx = hsa_config.get('wrist_camera_idx', 1)  # Default: left_wrist (index 1)
            self.tactile_camera_idx = hsa_config.get('tactile_camera_idx', 0)  # Default: first tactile sensor
            
            # Flag to track if we've warned about missing third-person camera params
            self._warned_tp_params = False
            
            # Initialize camera_params if not provided
            if self.camera_params is None:
                self.camera_params = {}
            
            # K_tp: Intrinsic matrix (from camera specs or calibration)
            # E_tp: Extrinsic matrix (camera pose in robot base frame)
            self.camera_params['K_tp'] = np.array([[647.0, 0.0, 653.0], 
                                                   [0.0, 644.0, 364.0], 
                                                   [0.0, 0.0, 1.0]])
            self.camera_params['E_tp'] = np.array([[1.0, 0.0, 0.0, 0.7], 
                                                   [0.0, -1.0, 0.0, -0.49], 
                                                   [0.0, 0.0, 1.0, 1.14], 
                                                   [0.0, 0.0, 0.0, 1.0]])
            
            # Print HSA and CLIP configuration
            print(f"\n{'='*60}")
            print(f"HSA Loss Configuration")
            print(f"{'='*60}")
            print(f"  HSA Weight: {self.hsa_weight}")
            print(f"  Temperature: {temperature}")
            print(f"  Use Third-Person: {use_third_person}")
            print(f"  Robot Type: {self.robot_type}")
            print(f"  Wrist Camera Index: {self.wrist_camera_idx} (left_wrist)")
            print(f"  Tactile Camera Index: {self.tactile_camera_idx}")
            print(f"  Third-Person Camera: Enabled (using hardcoded calibration)")
            
            # Print CLIP backbone parameters
            print(f"\n{'='*60}")
            print(f"CLIP Backbone Parameters")
            print(f"{'='*60}")
            clip_total_params = sum(p.numel() for p in self.feature_extractor.backbone.parameters())
            clip_trainable_params = sum(p.numel() for p in self.feature_extractor.backbone.parameters() if p.requires_grad)
            clip_frozen_params = clip_total_params - clip_trainable_params
            num_param_tensors = len(list(self.feature_extractor.backbone.parameters()))
            
            print(f"  Parameter Tensors: {num_param_tensors}")
            print(f"  Total Parameters: {clip_total_params:,} ({clip_total_params/1e6:.1f}M)")
            print(f"  Trainable Parameters: {clip_trainable_params:,} ({clip_trainable_params/1e6:.1f}M)")
            print(f"  Frozen Parameters: {clip_frozen_params:,}")
            print(f"  Image Size: {img_size}x{img_size}")
            print(f"  Patch Size: {patch_size}x{patch_size}")
            print(f"  Embed Dimension: {feature_dim}")
            print(f"  Feature Grid Shape: {self.feature_extractor.feature_grid_shape}")
            
            # Verify all CLIP components are included
            print(f"\n  CLIP Module Breakdown:")
            for name, module in self.feature_extractor.backbone.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"    {name}: {module_params:,} params ({module_params/1e6:.1f}M, {module_trainable:,} trainable)")
            
            print(f"{'='*60}\n")
        else:
            self.feature_extractor = None
            self.hsa_loss_fn = None
            print("HSA Loss disabled")
    
    def extract_tactile_visual_features(self,
                                        wrist_image: torch.Tensor,
                                        tactile_image: torch.Tensor,
                                        qpos: torch.Tensor,
                                        tp_image: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract tactile and visual features using the feature extractor.
        
        Args:
            wrist_image: Wrist camera image, shape (B, C, H, W) - mounted on robot
            tactile_image: Tactile sensor image, shape (B, C, H, W)
            qpos: Joint angles, shape (B, state_dim) - first 6 are joint angles (only needed for third-person)
            tp_image: Optional third-person camera image, shape (B, C, H, W) - stationary in workspace
        
        Returns:
            h_tau: Tactile features (B, D)
            h_w: Wrist visual features (B, D)
            h_tp: Third-person features (B, D) if tp_image provided, else None
        """
        B = wrist_image.shape[0]
        device = wrist_image.device
        
        h_tau_list = []
        h_w_list = []
        h_tp_list = [] if tp_image is not None else None
        
        for i in range(B):
            # Extract gripper width from qpos[6] (ONE side position, not total gap)
            gripper_width = qpos[i, 6].cpu().item()  # in meters
            
            # Convert images to numpy (H, W, C) format
            wrist_img_np = wrist_image[i].permute(1, 2, 0).cpu().numpy()
            wrist_img_np = (wrist_img_np * 255).astype(np.uint8)
            
            tactile_img_np = tactile_image[i].permute(1, 2, 0).cpu().numpy()
            tactile_img_np = (tactile_img_np * 255).astype(np.uint8)
            
            # Get wrist camera intrinsics if available
            K_wrist = None
            if self.camera_params is not None and 'K_wrist' in self.camera_params:
                K_wrist = self.camera_params['K_wrist']
            
            # Prepare third-person image if provided
            tp_img_np = None
            bbox_tp = None
            if tp_image is not None:
                tp_img_np = tp_image[i].permute(1, 2, 0).cpu().numpy()
                tp_img_np = (tp_img_np * 255).astype(np.uint8)
                
                # Only compute bounding box for third-person camera (stationary)
                # Wrist camera uses gripper-aware offset instead
                joint_angles = qpos[i, :6].cpu().numpy()
                
                # Compute sensor pose using forward kinematics
                sensor_pose = ForwardKinematics.compute_tactile_sensor_pose(
                    joint_angles=joint_angles,
                    robot_type=self.robot_type,
                    sensor_offset=self.sensor_offset
                )
                
                # Require real camera parameters for third-person camera
                if self.camera_params is not None and 'K_tp' in self.camera_params and 'E_tp' in self.camera_params:
                    K_tp = self.camera_params['K_tp']
                    E_tp = self.camera_params['E_tp']
                    
                    # Compute bounding box for third-person view
                    tp_h, tp_w = tp_img_np.shape[:2]
                    bbox_tp = CameraProjection.compute_sensor_bounding_box(
                        sensor_pose=sensor_pose,
                        sensor_size=self.sensor_size,
                        K=K_tp,
                        E=E_tp,
                        img_size=(tp_w, tp_h)
                    )
                else:
                    # Skip third-person processing if real camera params not provided
                    if not self._warned_tp_params:
                        print("Warning: Third-person camera parameters not provided, skipping third-person features")
                        self._warned_tp_params = True
                    tp_img_np = None
                    bbox_tp = None
            
            # Extract features with gripper-aware offset
            features = self.feature_extractor.extract_features(
                wrist_image=wrist_img_np,
                tactile_image=tactile_img_np,
                gripper_width=gripper_width,
                camera_K_wrist=K_wrist,
                tp_image=tp_img_np,
                bbox_tp=bbox_tp
            )
            
            h_tau_list.append(features['h_tau'])
            h_w_list.append(features['h_w'])
            if tp_image is not None and 'h_tp' in features:
                h_tp_list.append(features['h_tp'])
        
        # Stack into batch
        h_tau = torch.stack(h_tau_list).to(device)
        h_w = torch.stack(h_w_list).to(device)
        h_tp = torch.stack(h_tp_list).to(device) if h_tp_list else None
        
        return h_tau, h_w, h_tp
    
    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        """
        Forward pass with optional HSA loss computation.
        
        Args:
            qpos: Joint positions/states, shape (B, state_dim)
            image: Camera images as list [rgb_tensor, tactile_tensor]
                   rgb_tensor: (B, num_rgb_cameras, C, H, W)
                   tactile_tensor: (B, num_tactile_cameras, C, H, W)
            actions: Action sequences (for training), shape (B, seq_len, action_dim)
            is_pad: Padding mask, shape (B, seq_len)
            vq_sample: VQ sample (if using VQ-VAE)
        
        Returns:
            If training: loss_dict with keys: 'l1', 'kl', 'loss', optionally 'hsa_wrist', 'hsa_total'
            If inference: predicted actions
        """
        # IMPORTANT: Save unnormalized images BEFORE base policy normalizes them
        # Base policy applies ImageNet normalization which would corrupt uint8 conversion
        if self.enable_hsa and self.training and actions is not None:
            if isinstance(image, list) and len(image) >= 2:
                # Clone to avoid modifying original (base policy will normalize its copy)
                rgb_images_unnorm = image[0].clone()  # (B, num_rgb, C, H, W) 
                tactile_images_unnorm = image[1].clone()  # (B, num_tactile, C, H, W)
            else:
                rgb_images_unnorm = None
                tactile_images_unnorm = None
        else:
            rgb_images_unnorm = None
            tactile_images_unnorm = None
        
        # Call base ACTJEPAAdapter policy (this will normalize images)
        base_output = super().__call__(qpos, image, actions, is_pad, vq_sample)
        
        # If not training or HSA not enabled, return base output
        if actions is None:  # Inference mode
            return base_output
        
        # Determine whether to compute HSA
        compute_hsa = self.enable_hsa and self.training
        
        if not compute_hsa:
            return base_output
        
        # Use unnormalized images for HSA (in [0,1] range, not ImageNet normalized)
        if rgb_images_unnorm is None or tactile_images_unnorm is None:
            print("Warning: HSA requires image as list [rgb_tensor, tactile_tensor], skipping HSA loss")
            return base_output
        
        rgb_images = rgb_images_unnorm  # (B, num_rgb, C, H, W) - RGB cameras: [top, left_wrist, right_wrist]
        tactile_images = tactile_images_unnorm  # (B, num_tactile, C, H, W)
        
        # Select specific camera indices
        # self.wrist_camera_idx = 1 by default (left_wrist)
        wrist_image = rgb_images[:, self.wrist_camera_idx]  # (B, C, H, W)
        tactile_img = tactile_images[:, self.tactile_camera_idx]  # (B, C, H, W)
        
        # Extract features and compute HSA loss
        try:
            # rgb_images has [top, left_wrist, right_wrist]
            tp_image = None
            if rgb_images.shape[1] >= 3:  # Have 3+ cameras, first is top (third-person)
                tp_image = rgb_images[:, 0]  # Use first RGB camera (index 0 = top) as third-person
            
            h_tau, h_w, h_tp = self.extract_tactile_visual_features(
                wrist_image=wrist_image,
                tactile_image=tactile_img,
                qpos=qpos,
                tp_image=tp_image
            )
            
            # Compute HSA loss
            hsa_loss_dict = self.hsa_loss_fn(h_tau=h_tau, h_w=h_w, h_tp=h_tp)
            
            # Add HSA loss to total loss
            base_output['hsa_wrist'] = hsa_loss_dict['hsa_wrist']
            base_output['hsa_total'] = hsa_loss_dict['hsa_total']
            if 'hsa_tp' in hsa_loss_dict:
                base_output['hsa_tp'] = hsa_loss_dict['hsa_tp']
            
            # IMPORTANT: Add weighted HSA loss to total loss for backprop
            base_output['loss'] = base_output['loss'] + self.hsa_weight * hsa_loss_dict['hsa_total']
            
            # Verify gradients will flow (check requires_grad)
            if not hasattr(self, '_grad_check_done'):
                self._grad_check_done = True
                print(f"\n[HSA Gradient Check]")
                print(f"  h_tau requires_grad: {h_tau.requires_grad}")
                print(f"  h_w requires_grad: {h_w.requires_grad}")
                print(f"  HSA loss requires_grad: {hsa_loss_dict['hsa_total'].requires_grad}")
                print(f"  Total loss requires_grad: {base_output['loss'].requires_grad}\n")
            
        except Exception as e:
            print(f"Warning: Failed to compute HSA loss: {e}")
            import traceback
            traceback.print_exc()
            # Continue training without HSA loss on error
        
        return base_output
    
    def train(self, mode=True):
        """
        Set the module in training mode.
        Also controls the feature extractor backbone if HSA is enabled.
        """
        super().train(mode)
        if self.enable_hsa and self.feature_extractor is not None:
            if mode:
                self.feature_extractor.backbone.train()
            else:
                self.feature_extractor.backbone.eval()
        return self
    
    def eval(self):
        """
        Set the module in evaluation mode.
        """
        return self.train(False)
    
    def configure_optimizers(self):
        """
        Configure optimizer. If HSA is enabled, include feature extractor parameters.
        """
        base_optimizer = super().configure_optimizers()
        
        if self.enable_hsa and self.feature_extractor is not None:
            # Get parameters from base optimizer
            all_params = list(base_optimizer.param_groups)
            
            # Add feature extractor parameters
            # Higher LR helps HSA loss converge faster (CLIP needs to learn alignment from scratch)
            feature_params = list(self.feature_extractor.backbone.parameters())
            
            # Verify parameters are trainable
            trainable_params = [p for p in feature_params if p.requires_grad]
            frozen_params = [p for p in feature_params if not p.requires_grad]
            
            if len(trainable_params) > 0:
                clip_lr_multiplier = 1.0  # Same as base LR for faster convergence
                all_params.append({
                    'params': trainable_params,
                    'lr': base_optimizer.param_groups[0]['lr'] * clip_lr_multiplier
                })
                print(f"Added {len(trainable_params)} CLIP params to optimizer with LR={base_optimizer.param_groups[0]['lr'] * clip_lr_multiplier:.2e}")
                if len(frozen_params) > 0:
                    print(f"  ⚠ WARNING: {len(frozen_params)} CLIP params are frozen!")
            else:
                print(f"  ⚠ ERROR: No trainable CLIP parameters found! All {len(feature_params)} params are frozen!")
            
            # Create new optimizer with all parameters
            optimizer = torch.optim.AdamW(
                all_params,
                lr=base_optimizer.param_groups[0]['lr']
            )
            return optimizer
        
        return base_optimizer


def create_default_hsa_config(
    enable_hsa: bool = True,
    hsa_weight: float = 1.0,
    temperature: float = 0.07,
    img_size: int = 224,
    feature_dim: int = 768,
    num_heads: int = 12,
    wrist_camera_idx: int = 0,
    tactile_camera_idx: int = 0,
    robot_type: str = 'Nova 2'
) -> Dict:
    """
    Create a default HSA configuration dictionary.
    
    Args:
        enable_hsa: Whether to enable HSA loss
        hsa_weight: Weight for HSA loss
        temperature: Temperature for contrastive loss
        img_size: Image size for feature extraction
        feature_dim: Feature dimension (768 for ViT-L, 1408 for ViT-G)
        num_heads: Number of attention heads (12 for ViT-L, 16 for ViT-G)
        wrist_camera_idx: Index of wrist camera in RGB camera list
        tactile_camera_idx: Index of tactile sensor in tactile camera list
        robot_type: Robot type for forward kinematics
    
    Returns:
        HSA configuration dictionary
    """
    return {
        'enable_hsa': enable_hsa,
        'hsa_weight': hsa_weight,
        'temperature': temperature,
        'use_third_person': False,
        'tp_weight': 0.5,
        'feature_dim': feature_dim,
        'num_heads': num_heads,
        'img_size': img_size,
        'patch_size': 16,
        'camera_params': None,  # Will use default
        'robot_type': robot_type,
        'sensor_offset': np.array([0.0, 0.0, 0.02]),
        'sensor_size': (0.04, 0.04),
        'wrist_camera_idx': wrist_camera_idx,
        'tactile_camera_idx': tactile_camera_idx,
    }

