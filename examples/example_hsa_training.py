r"""
Example: Training ACT Policy with HSA Loss

This example demonstrates how to train an ACT policy with HSA (Hard Sample Aware)
loss for tactile-visual feature alignment.

The HSA loss aligns tactile sensor features with wrist camera features using
contrastive learning:

$$
\mathcal{L}_{\text{HSA-W}} = -\log \frac{\exp(h_\tau \cdot h_w / \kappa)}{\exp(h_\tau \cdot h_w / \kappa) + \sum_{i=1}^{N_k} \exp(h_\tau \cdot h_{w,i}^{\text{neg}} / \kappa)}
$$

Usage:
    # Basic training with HSA loss
    python examples/example_hsa_training.py --enable_hsa
    
    # With custom HSA weight
    python examples/example_hsa_training.py --enable_hsa --hsa_weight 2.0
    
    # With custom temperature
    python examples/example_hsa_training.py --enable_hsa --hsa_temperature 0.1
"""

import sys
import os
import torch
import numpy as np

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from ModelTrain.module.policy_with_hsa import ACTPolicyWithHSA, create_default_hsa_config


def example_basic_hsa_loss():
    """Example 1: Basic HSA loss computation."""
    print("\n" + "="*70)
    print("Example 1: Basic HSA Loss Computation")
    print("="*70)
    
    from dobot_control.hsa_loss import HSALoss
    
    # Create fake features
    batch_size = 8
    feature_dim = 768
    
    h_tau = torch.randn(batch_size, feature_dim)  # Tactile features
    h_w = torch.randn(batch_size, feature_dim)    # Wrist visual features
    
    # Create HSA loss
    hsa_loss_fn = HSALoss(temperature=0.07)
    
    # Compute loss (uses in-batch negatives)
    loss = hsa_loss_fn(h_tau, h_w)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  HSA Loss: {loss.item():.4f}")
    print(f"  Number of negative samples: {batch_size - 1} (in-batch)")
    
    return loss


def example_hsa_with_hard_negatives():
    """Example 2: HSA loss with explicit hard negatives."""
    print("\n" + "="*70)
    print("Example 2: HSA Loss with Hard Negatives")
    print("="*70)
    
    from dobot_control.hsa_loss import HSALoss
    
    batch_size = 8
    feature_dim = 768
    num_hard_neg = 10
    
    h_tau = torch.randn(batch_size, feature_dim)
    h_w = torch.randn(batch_size, feature_dim)
    
    # Create hard negatives (e.g., from a memory bank or previous batches)
    hard_negatives = torch.randn(batch_size, num_hard_neg, feature_dim)
    
    hsa_loss_fn = HSALoss(temperature=0.07)
    loss = hsa_loss_fn(h_tau, h_w, hard_negatives)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Number of hard negatives per sample: {num_hard_neg}")
    print(f"  HSA Loss: {loss.item():.4f}")
    
    return loss


def example_feature_extraction():
    """Example 3: Tactile-visual feature extraction."""
    print("\n" + "="*70)
    print("Example 3: Tactile-Visual Feature Extraction")
    print("="*70)
    
    from dobot_control.tactile_feature_extraction import (
        TactileFeatureExtractor,
        ForwardKinematics,
        CameraProjection,
        generate_fake_images,
        generate_fake_camera_params
    )
    
    # Generate fake data
    img_size = (640, 480)
    wrist_image, tactile_image = generate_fake_images(img_size)
    K_w, E_w = generate_fake_camera_params(img_size)
    
    # Joint angles
    joint_angles = np.array([0.1, 0.2, -0.3, 0.4, -0.1, 0.2])
    sensor_offset = np.array([0.0, 0.0, 0.02])
    
    # Compute sensor pose
    sensor_pose = ForwardKinematics.compute_tactile_sensor_pose(
        joint_angles=joint_angles,
        robot_type="Nova 2",
        sensor_offset=sensor_offset
    )
    
    # Compute bounding box
    sensor_size = (0.04, 0.04)
    bbox_w = CameraProjection.compute_sensor_bounding_box(
        sensor_pose=sensor_pose,
        sensor_size=sensor_size,
        K=K_w,
        E=E_w,
        img_size=img_size
    )
    
    # Extract features
    extractor = TactileFeatureExtractor(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        device='cpu'
    )
    
    features = extractor.extract_features(
        wrist_image=wrist_image,
        tactile_image=tactile_image,
        bbox_wrist=bbox_w
    )
    
    h_tau = features['h_tau']
    h_w = features['h_w']
    
    print(f"  Wrist image shape: {wrist_image.shape}")
    print(f"  Tactile image shape: {tactile_image.shape}")
    print(f"  Sensor pose: {sensor_pose[:3, 3]}")  # Position
    print(f"  Bounding box: x=[{bbox_w['x_min']:.0f}, {bbox_w['x_max']:.0f}], "
          f"y=[{bbox_w['y_min']:.0f}, {bbox_w['y_max']:.0f}]")
    print(f"  h_tau shape: {h_tau.shape}")
    print(f"  h_w shape: {h_w.shape}")
    
    return h_tau, h_w


def example_act_policy_with_hsa():
    """Example 4: Creating ACT policy with HSA loss."""
    print("\n" + "="*70)
    print("Example 4: ACT Policy with HSA Loss")
    print("="*70)
    
    # ACT configuration
    act_config = {
        'lr': 1e-5,
        'num_queries': 100,
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': ['top', 'left_wrist', 'right_wrist'],
        'tactile_camera_names': ['tactile_left'],
        'vq': False,
        'vq_class': None,
        'vq_dim': None,
        'action_dim': 16,
        'no_encoder': False,
        'use_vitg': False,
        'vitg_ckpt_path': None
    }
    
    # HSA configuration
    hsa_config = create_default_hsa_config(
        enable_hsa=True,
        hsa_weight=1.0,
        temperature=0.07,
        img_size=224,
        wrist_camera_name='left_wrist',
        camera_names=['top', 'left_wrist', 'right_wrist'],
        robot_type='Nova 2'
    )
    
    print("  ACT Configuration:")
    print(f"    - Hidden dim: {act_config['hidden_dim']}")
    print(f"    - Num queries: {act_config['num_queries']}")
    print(f"    - KL weight: {act_config['kl_weight']}")
    
    print("\n  HSA Configuration:")
    print(f"    - Enabled: {hsa_config['enable_hsa']}")
    print(f"    - HSA weight: {hsa_config['hsa_weight']}")
    print(f"    - Temperature: {hsa_config['temperature']}")
    print(f"    - Feature dim: {hsa_config['feature_dim']}")
    print(f"    - Image size: {hsa_config['img_size']}")
    print(f"    - Robot type: {hsa_config['robot_type']}")
    
    # Create policy
    try:
        policy = ACTPolicyWithHSA(act_config, hsa_config)
        print("\n  ✓ Policy created successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return policy
    except Exception as e:
        print(f"\n  ✗ Failed to create policy: {e}")
        return None


def example_training_script():
    """Example 5: Training script usage."""
    print("\n" + "="*70)
    print("Example 5: Training Script Usage")
    print("="*70)
    
    print("\n  To train with HSA loss, use the enhanced training script:")
    print("\n  Basic usage:")
    print("  ```bash")
    print("  python ModelTrain/model_train.py \\")
    print("      --task_name dobot_pick_random_1013 \\")
    print("      --ckpt_dir ./ckpt/dobot_pick_hsa \\")
    print("      --enable_hsa \\")
    print("      --batch_size 16 \\")
    print("      --num_steps 30000")
    print("  ```")
    
    print("\n  With custom HSA parameters:")
    print("  ```bash")
    print("  python ModelTrain/model_train.py \\")
    print("      --task_name dobot_pick_random_1013 \\")
    print("      --ckpt_dir ./ckpt/dobot_pick_hsa \\")
    print("      --enable_hsa \\")
    print("      --hsa_weight 2.0 \\")
    print("      --hsa_temperature 0.1 \\")
    print("      --hsa_img_size 224 \\")
    print("      --hsa_feature_dim 768 \\")
    print("      --robot_type 'Nova 2' \\")
    print("      --wrist_camera left_wrist \\")
    print("      --batch_size 16 \\")
    print("      --num_steps 30000")
    print("  ```")
    
    print("\n  Key HSA parameters:")
    print("    --enable_hsa              Enable HSA loss")
    print("    --hsa_weight FLOAT        Weight for HSA loss (default: 1.0)")
    print("    --hsa_temperature FLOAT   Temperature for contrastive loss (default: 0.07)")
    print("    --hsa_img_size INT        Image size for feature extraction (default: 224)")
    print("    --hsa_feature_dim INT     Feature dimension (default: 768)")
    print("    --robot_type STR          Robot type: 'Nova 2' or 'Nova 5' (default: 'Nova 2')")
    print("    --wrist_camera STR        Wrist camera name (default: 'left_wrist')")


def example_loss_interpretation():
    """Example 6: Understanding HSA loss values."""
    print("\n" + "="*70)
    print("Example 6: Understanding HSA Loss Values")
    print("="*70)
    
    from dobot_control.hsa_loss import HSALoss
    
    feature_dim = 768
    temperature = 0.07
    hsa_loss_fn = HSALoss(temperature=temperature)
    
    print("\n  Scenario 1: Perfect alignment (h_tau = h_w)")
    h_tau = torch.randn(8, feature_dim)
    h_w = h_tau.clone()  # Perfect match
    loss1 = hsa_loss_fn(h_tau, h_w)
    print(f"    Loss: {loss1.item():.4f} (lower is better)")
    
    print("\n  Scenario 2: Moderate alignment")
    h_w = h_tau + 0.5 * torch.randn(8, feature_dim)  # Some noise
    loss2 = hsa_loss_fn(h_tau, h_w)
    print(f"    Loss: {loss2.item():.4f}")
    
    print("\n  Scenario 3: Poor alignment")
    h_w = torch.randn(8, feature_dim)  # Random features
    loss3 = hsa_loss_fn(h_tau, h_w)
    print(f"    Loss: {loss3.item():.4f} (higher indicates misalignment)")
    
    print("\n  Typical training behavior:")
    print("    - Initial loss: 4.0 - 6.0 (random features)")
    print("    - Mid-training: 2.0 - 3.0 (partial alignment)")
    print("    - Well-trained: 0.5 - 1.5 (good alignment)")
    print("    - Perfect: ~0.0 (rarely achieved with real data)")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("HSA Loss Training Examples")
    print("="*70)
    print("\nThese examples demonstrate how to use HSA (Hard Sample Aware) loss")
    print("for aligning tactile and visual features in ACT policy training.")
    
    # Run examples
    example_basic_hsa_loss()
    example_hsa_with_hard_negatives()
    example_feature_extraction()
    example_act_policy_with_hsa()
    example_training_script()
    example_loss_interpretation()
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("\nHSA Loss combines tactile-visual feature alignment with ACT training:")
    print("  Total Loss = L1_loss + KL_weight × KL_loss + HSA_weight × HSA_loss")
    print("\nBenefits:")
    print("  ✓ Better tactile-visual correspondence")
    print("  ✓ Improved manipulation with tactile feedback")
    print("  ✓ More robust feature representations")
    print("\nNext steps:")
    print("  1. Prepare dataset with tactile sensor data")
    print("  2. Configure camera parameters and robot type")
    print("  3. Run training with --enable_hsa flag")
    print("  4. Monitor HSA loss alongside ACT losses")
    print("  5. Evaluate policy on real robot tasks")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

