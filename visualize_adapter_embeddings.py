#!/usr/bin/env python3
"""
Visualize JEPA-Adapter Feature Embeddings as Heatmaps

This script loads a trained ACTJEPAAdapter model, runs a single forward pass
on a dataset sample, and extracts the adapter output embeddings to visualize
as a heatmap.

Usage:
    python visualize_adapter_embeddings.py \
        --ckpt_dir ./ckpt/actjepa_hsa_peg_1107 \
        --ckpt_name policy_last.ckpt \
        --vit_ckpt_path ./jepa_ckpt/vitl.pt \
        --sample_idx 0 \
        --output adapter_embeddings.png
"""

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'ModelTrain'))
sys.path.append(os.path.join(BASE_DIR, 'ModelTrain/detr'))
sys.path.append(os.path.join(BASE_DIR, 'robomimic-r2d2'))

import argparse
import pickle
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from torchvision import transforms

from ModelTrain.module.policy_jepa_adapter_with_hsa import ACTJEPAHsa
from ModelTrain.constants import TASK_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize JEPA-Adapter embeddings')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Path to checkpoint directory (e.g., ./ckpt/actjepa_hsa_peg_1107)')
    parser.add_argument('--ckpt_name', type=str, default='policy_last.ckpt',
                        help='Checkpoint filename')
    parser.add_argument('--vit_ckpt_path', type=str, default=None,
                        help='Path to ViT checkpoint (overrides config if provided)')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Dataset sample index to visualize')
    parser.add_argument('--episode_idx', type=int, default=0,
                        help='Episode index in dataset')
    parser.add_argument('--timestep', type=int, default=50,
                        help='Timestep within episode to visualize')
    parser.add_argument('--output', type=str, default='adapter_embeddings.png',
                        help='Output filename for heatmap')
    parser.add_argument('--reduction', type=str, default='mean', choices=['mean', 'pca', 'std'],
                        help='Method to reduce high-dim embeddings to scalar')
    return parser.parse_args()


def load_checkpoint_config(ckpt_dir):
    """Load config and dataset stats from checkpoint directory."""
    ckpt_path = Path(ckpt_dir)
    
    config_path = ckpt_path / 'config.pkl'
    stats_path = ckpt_path / 'dataset_stats.pkl'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Dataset stats not found: {stats_path}")
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    print(f"Loaded config from {config_path}")
    print(f"Policy class: {config.get('policy_class', 'Unknown')}")
    print(f"Camera names: {config.get('camera_names', [])}")
    print(f"Tactile camera names: {config.get('policy_config', {}).get('tactile_camera_names', [])}")
    
    return config, stats


def load_model(config, ckpt_dir, ckpt_name, vit_ckpt_override=None):
    """Load trained model from checkpoint."""
    policy_config = config['policy_config']
    
    # Override ViT checkpoint if provided
    if vit_ckpt_override:
        print(f"Overriding ViT checkpoint: {vit_ckpt_override}")
        policy_config['vitg_ckpt_path'] = vit_ckpt_override
        policy_config['vit_ckpt_path'] = vit_ckpt_override
    
    # Print ViT checkpoint being used
    vit_path = policy_config.get('vitg_ckpt_path') or policy_config.get('vit_ckpt_path')
    print(f"Using ViT checkpoint: {vit_path}")
    
    # Create policy (HSA disabled for inference)
    hsa_config = {'enable_hsa': False}
    policy = ACTJEPAHsa(policy_config, hsa_config)
    
    # Load checkpoint weights
    ckpt_path = Path(ckpt_dir) / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cuda')
    
    # If ViT override is provided, filter out ViT encoder weights from policy checkpoint
    # This prevents the policy checkpoint from overwriting the new ViT weights
    if vit_ckpt_override:
        print("Filtering out ViT encoder weights from policy checkpoint to use override...")
        original_keys = len(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if 'vitg_base.encoder' not in k}
        filtered_keys = original_keys - len(state_dict)
        print(f"Filtered {filtered_keys} ViT encoder keys (kept {len(state_dict)} adapter/other keys)")
    
    policy.model.load_state_dict(state_dict, strict=False)
    policy.model.eval()
    
    print("Model loaded successfully!")
    return policy


def load_dataset_sample(config, stats, episode_idx=0, timestep=50):
    """Load a single sample from the dataset."""
    # Get dataset directory from task config
    task_name = config.get('task_name')
    if task_name and task_name in TASK_CONFIGS:
        dataset_dir = TASK_CONFIGS[task_name]['dataset_dir']
    else:
        dataset_dir = config.get('dataset_dir')
    
    if isinstance(dataset_dir, list):
        dataset_dir = dataset_dir[0]
    
    if not dataset_dir:
        raise ValueError("Could not determine dataset directory from config")
    
    print(f"Loading data from: {dataset_dir}")
    
    # Find all HDF5 files
    dataset_path = Path(dataset_dir)
    hdf5_files = sorted(dataset_path.glob('episode_*.hdf5'))
    
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")
    
    if episode_idx >= len(hdf5_files):
        raise ValueError(f"Episode index {episode_idx} out of range (0-{len(hdf5_files)-1})")
    
    episode_file = hdf5_files[episode_idx]
    print(f"Loading episode: {episode_file}")
    
    camera_names = config.get('camera_names', [])
    tactile_camera_names = config.get('policy_config', {}).get('tactile_camera_names', [])
    
    with h5py.File(episode_file, 'r') as f:
        # Check episode length
        episode_len = f['/action'].shape[0]
        if timestep >= episode_len:
            print(f"Warning: timestep {timestep} >= episode_len {episode_len}, using timestep 0")
            timestep = 0
        
        # Load qpos
        qpos = f['/observations/qpos'][timestep]
        
        # Load images
        image_dict = {}
        
        # Load RGB cameras
        for cam_name in camera_names:
            if f"/observations/images/{cam_name}" in f:
                img_data = f[f"/observations/images/{cam_name}"][timestep]
                # Check if compressed
                if img_data.dtype == np.uint8 and len(img_data.shape) == 1:
                    img_data = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                image_dict[cam_name] = img_data
        
        # Load tactile sensors
        for cam_name in tactile_camera_names:
            if f"/observations/{cam_name}" in f:
                img_data = f[f"/observations/{cam_name}"][timestep]
                # Check if compressed
                if img_data.dtype == np.uint8 and len(img_data.shape) == 1:
                    img_data = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                image_dict[cam_name] = img_data
    
    # Normalize qpos
    if 'qpos_mean' in stats and 'qpos_std' in stats:
        qpos = (qpos - stats['qpos_mean']) / stats['qpos_std']
    
    print(f"Loaded sample - timestep: {timestep}, qpos shape: {qpos.shape}")
    print(f"Images loaded: {list(image_dict.keys())}")
    
    return qpos, image_dict, camera_names, tactile_camera_names


def preprocess_images(image_dict, camera_names, tactile_camera_names):
    """Preprocess images for model input."""
    # Separate RGB and tactile
    rgb_cameras = [cam for cam in camera_names if cam not in tactile_camera_names]
    
    # Process RGB images
    rgb_images = []
    for cam_name in rgb_cameras:
        img = image_dict[cam_name]  # (H, W, C) in BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = img.transpose(2, 0, 1)  # (C, H, W)
        rgb_images.append(img)
    
    rgb_stacked = np.stack(rgb_images, axis=0)  # (num_rgb, C, H, W)
    rgb_tensor = torch.from_numpy(rgb_stacked / 255.0).float().cuda().unsqueeze(0)  # (1, num_rgb, C, H, W)
    
    # Process tactile images (resize to 224x224 for ViT)
    tactile_images = []
    for cam_name in tactile_camera_names:
        img = image_dict[cam_name]  # (H, W, C) in BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (224, 224))  # Resize for ViT
        img = img.transpose(2, 0, 1)  # (C, H, W)
        tactile_images.append(img)
    
    tactile_stacked = np.stack(tactile_images, axis=0)  # (num_tactile, C, H, W)
    tactile_tensor = torch.from_numpy(tactile_stacked / 255.0).float().cuda().unsqueeze(0)  # (1, num_tactile, C, H, W)
    
    # Return as list (different resolutions)
    return [rgb_tensor, tactile_tensor], tactile_images[0] if tactile_images else None


class EmbeddingExtractor:
    """Hook to extract embeddings from the adapter."""
    def __init__(self):
        self.embeddings = None
    
    def hook_fn(self, module, input, output):
        """Hook function to capture adapter output."""
        # The output of patch_adapter is the adapted_patches
        # Shape: (B, num_patches, embed_dim)
        self.embeddings = output.detach().cpu()
    
    def register_hook(self, model):
        """Register hook on the adapter layer."""
        # Navigate to the adapter in the model hierarchy
        # model -> model.vitg_encoder_shared -> patch_adapter
        adapter = model.model.vitg_encoder_shared.patch_adapter
        self.handle = adapter.register_forward_hook(self.hook_fn)
        print(f"Registered hook on patch_adapter")
    
    def remove_hook(self):
        """Remove the hook."""
        if hasattr(self, 'handle'):
            self.handle.remove()


def extract_embeddings(policy, qpos, image_list):
    """Run forward pass and extract adapter embeddings."""
    # Create extractor
    extractor = EmbeddingExtractor()
    extractor.register_hook(policy)
    
    # Prepare input
    qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # (1, state_dim)
    
    # Run forward pass
    with torch.no_grad():
        _ = policy(qpos_tensor, image_list)
    
    # Get embeddings
    embeddings = extractor.embeddings
    extractor.remove_hook()
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings


def create_heatmap(embeddings, output_path, reduction='mean'):
    """Create and save heatmap visualization."""
    # embeddings shape: (B, num_patches, embed_dim)
    # We want to reduce to (num_patches,) for heatmap
    
    embeddings = embeddings.squeeze(0)  # (num_patches, embed_dim)
    num_patches, embed_dim = embeddings.shape
    
    print(f"Processing embeddings: {num_patches} patches, {embed_dim} dimensions")
    
    # Reduce dimensionality to scalar per patch
    if reduction == 'mean':
        # Mean across embedding dimension
        heatmap_values = embeddings.mean(dim=1).numpy()
    elif reduction == 'std':
        # Standard deviation across embedding dimension
        heatmap_values = embeddings.std(dim=1).numpy()
    elif reduction == 'pca':
        # First principal component
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        heatmap_values = pca.fit_transform(embeddings.numpy()).flatten()
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")
    
    # Determine grid size (assume square patches)
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        print(f"Warning: num_patches={num_patches} is not a perfect square")
        # Pad to next square
        target_size = int(np.ceil(np.sqrt(num_patches))) ** 2
        heatmap_values = np.pad(heatmap_values, (0, target_size - num_patches), constant_values=0)
        grid_size = int(np.sqrt(target_size))
    
    # Reshape to 2D grid
    heatmap_2d = heatmap_values.reshape(grid_size, grid_size)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_2d, 
                cmap='viridis', 
                square=True, 
                cbar_kws={'label': f'Embedding {reduction}'},
                ax=ax)
    
    ax.set_title(f'Adapter Feature Embeddings ({grid_size}x{grid_size})', fontsize=14)
    ax.set_xlabel('Patch X')
    ax.set_ylabel('Patch Y')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    
    # Also save a version with the tactile image side by side if available
    plt.close()
    
    return heatmap_2d


def main():
    args = parse_args()
    
    print("=" * 60)
    print("JEPA-Adapter Embedding Visualization")
    print("=" * 60)
    
    # Load config
    config, stats = load_checkpoint_config(args.ckpt_dir)
    
    # Load model
    policy = load_model(config, args.ckpt_dir, args.ckpt_name, args.vit_ckpt_path)
    
    # Load dataset sample
    qpos, image_dict, camera_names, tactile_camera_names = load_dataset_sample(
        config, stats, args.episode_idx, args.timestep
    )
    
    # Preprocess images
    image_list, tactile_img = preprocess_images(image_dict, camera_names, tactile_camera_names)
    
    # Extract embeddings
    embeddings = extract_embeddings(policy, qpos, image_list)
    
    # Create heatmap
    heatmap_2d = create_heatmap(embeddings, args.output, args.reduction)
    
    # Create a combined visualization with tactile image and heatmap
    if tactile_img is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Show original tactile image
        axes[0].imshow(tactile_img.transpose(1, 2, 0))
        axes[0].set_title('Input Tactile Image')
        axes[0].axis('off')
        
        # Show heatmap
        sns.heatmap(heatmap_2d, 
                    cmap='viridis', 
                    square=True, 
                    cbar_kws={'label': f'Embedding {args.reduction}'},
                    ax=axes[1])
        axes[1].set_title(f'Adapter Embeddings ({heatmap_2d.shape[0]}x{heatmap_2d.shape[1]})')
        axes[1].set_xlabel('Patch X')
        axes[1].set_ylabel('Patch Y')
        
        combined_output = args.output.replace('.png', '_combined.png')
        plt.tight_layout()
        plt.savefig(combined_output, dpi=150, bbox_inches='tight')
        print(f"Combined visualization saved to: {combined_output}")
        plt.close()
    
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

