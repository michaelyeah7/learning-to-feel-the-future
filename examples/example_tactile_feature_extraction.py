"""
Example script demonstrating tactile-visual feature extraction.

This script:
1. Generates fake wrist and tactile images
2. Generates fake camera parameters
3. Computes 3D pose of tactile sensor using forward kinematics
4. Projects the pose to 2D bounding boxes
5. Extracts features h_tau and h_w using CLIP-like backbone
"""

import numpy as np
import matplotlib.pyplot as plt
from dobot_control.tactile_feature_extraction import (
    TactileFeatureExtractor,
    ForwardKinematics,
    CameraProjection,
    generate_fake_camera_params,
    generate_fake_images
)


def main():
    """Main example function."""
    print("=" * 60)
    print("Tactile-Visual Feature Extraction Example")
    print("=" * 60)
    
    # ========== Step 1: Generate Fake Data ==========
    print("\n[Step 1] Generating fake images and camera parameters...")
    
    img_size = (640, 480)
    wrist_image, tactile_image = generate_fake_images(img_size)
    K_w, E_w = generate_fake_camera_params(img_size)  # Wrist camera params
    K_tp, E_tp = generate_fake_camera_params(img_size)  # Third-person camera params
    
    print(f"  - Wrist image shape: {wrist_image.shape}")
    print(f"  - Tactile image shape: {tactile_image.shape}")
    print(f"  - Camera intrinsic K:\n{K_w}")
    
    # ========== Step 2: Compute Tactile Sensor 3D Pose ==========
    print("\n[Step 2] Computing tactile sensor 3D pose using forward kinematics...")
    
    # Fake joint angles (in radians)
    joint_angles = np.array([
        0.1,   # q0
        0.2,   # q1
        -0.3,  # q2
        0.4,   # q3
        -0.1,  # q4
        0.2    # q5
    ])
    
    # Sensor offset from end-effector (in meters)
    sensor_offset = np.array([0.0, 0.0, 0.02])  # 2cm offset in z-direction
    
    # Compute sensor pose
    P_sensor = ForwardKinematics.compute_tactile_sensor_pose(
        joint_angles=joint_angles,
        robot_type="Nova 2",
        sensor_offset=sensor_offset
    )
    
    print(f"  - Joint angles (rad): {joint_angles}")
    print(f"  - Sensor pose (position): {P_sensor[:3, 3]}")
    print(f"  - Sensor pose matrix shape: {P_sensor.shape}")
    
    # ========== Step 3: Project 3D Pose to 2D Bounding Boxes ==========
    print("\n[Step 3] Projecting 3D pose to 2D bounding boxes...")
    
    # Define sensor size (width, height in meters)
    sensor_size = (0.04, 0.04)  # 4cm x 4cm sensor
    
    # Compute bounding box in wrist view
    bbox_w = CameraProjection.compute_sensor_bounding_box(
        sensor_pose=P_sensor,
        sensor_size=sensor_size,
        K=K_w,
        E=E_w,
        img_size=img_size
    )
    
    # Compute bounding box in third-person view
    bbox_tp = CameraProjection.compute_sensor_bounding_box(
        sensor_pose=P_sensor,
        sensor_size=sensor_size,
        K=K_tp,
        E=E_tp,
        img_size=img_size
    )
    
    print(f"  - Wrist bounding box: x=[{bbox_w['x_min']:.1f}, {bbox_w['x_max']:.1f}], "
          f"y=[{bbox_w['y_min']:.1f}, {bbox_w['y_max']:.1f}]")
    print(f"  - Third-person bounding box: x=[{bbox_tp['x_min']:.1f}, {bbox_tp['x_max']:.1f}], "
          f"y=[{bbox_tp['y_min']:.1f}, {bbox_tp['y_max']:.1f}]")
    
    # ========== Step 4: Extract Features ==========
    print("\n[Step 4] Extracting features using CLIP-like backbone...")
    
    # Initialize feature extractor
    extractor = TactileFeatureExtractor(
        img_size=640,  # Will resize images to this size
        patch_size=16,
        embed_dim=768,
        device='cpu'  # Use 'cuda' if GPU available
    )
    
    # Extract features
    features = extractor.extract_features(
        wrist_image=wrist_image,
        tactile_image=tactile_image,
        bbox_wrist=bbox_w,
        bbox_tp=bbox_tp
    )
    
    h_tau = features['h_tau']
    h_w = features['h_w']
    
    print(f"  - h_tau shape: {h_tau.shape}")
    print(f"  - h_w shape: {h_w.shape}")
    print(f"  - h_tau mean: {h_tau.mean().item():.4f}, std: {h_tau.std().item():.4f}")
    print(f"  - h_w mean: {h_w.mean().item():.4f}, std: {h_w.std().item():.4f}")
    
    # ========== Step 5: Visualize Results ==========
    print("\n[Step 5] Visualizing results...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot wrist image with bounding box
    axes[0, 0].imshow(wrist_image)
    rect_w = plt.Rectangle(
        (bbox_w['x_min'], bbox_w['y_min']),
        bbox_w['x_max'] - bbox_w['x_min'],
        bbox_w['y_max'] - bbox_w['y_min'],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    axes[0, 0].add_patch(rect_w)
    axes[0, 0].set_title('Wrist Camera Image\n(Red box: projected sensor)')
    axes[0, 0].axis('off')
    
    # Plot tactile image
    axes[0, 1].imshow(tactile_image)
    axes[0, 1].set_title('Tactile Sensor Image')
    axes[0, 1].axis('off')
    
    # Plot feature vectors
    axes[1, 0].plot(h_tau.detach().numpy())
    axes[1, 0].set_title('h_tau (Tactile Features)')
    axes[1, 0].set_xlabel('Feature Dimension')
    axes[1, 0].set_ylabel('Feature Value')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(h_w.detach().numpy())
    axes[1, 1].set_title('h_w (Wrist Visual Features)')
    axes[1, 1].set_xlabel('Feature Dimension')
    axes[1, 1].set_ylabel('Feature Value')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('tactile_feature_extraction_example.png', dpi=150)
    print("  - Visualization saved to 'tactile_feature_extraction_example.png'")
    
    # ========== Step 6: Demonstrate Feature Grid Mapping ==========
    print("\n[Step 6] Demonstrating feature grid mapping...")
    
    # Map bounding box to feature grid
    feat_x_min, feat_y_min, feat_x_max, feat_y_max = extractor.map_bbox_to_feature_grid(
        bbox_w, (wrist_image.shape[1], wrist_image.shape[0])
    )
    
    feat_h, feat_w = extractor.feature_grid_shape
    print(f"  - Original image size: {wrist_image.shape[1]} x {wrist_image.shape[0]}")
    print(f"  - Feature grid size: {feat_w} x {feat_h}")
    print(f"  - Original bbox: x=[{bbox_w['x_min']:.1f}, {bbox_w['x_max']:.1f}], "
          f"y=[{bbox_w['y_min']:.1f}, {bbox_w['y_max']:.1f}]")
    print(f"  - Feature grid bbox: x=[{feat_x_min}, {feat_x_max}], "
          f"y=[{feat_y_min}, {feat_y_max}]")
    print(f"  - Number of tokens selected: {(feat_x_max - feat_x_min + 1) * (feat_y_max - feat_y_min + 1)}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    # Show plot (optional)
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()

