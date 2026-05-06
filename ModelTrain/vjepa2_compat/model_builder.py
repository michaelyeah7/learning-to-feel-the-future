"""
Simple V-JEPA2 ViT-Giant model builder for Python 3.8 compatibility
"""

import torch
import torch.nn as nn


def create_vit_giant(img_size=224, patch_size=16, num_frames=2, tubelet_size=2):
    """
    Create a ViT-Giant model compatible with V-JEPA2 checkpoints.
    
    Args:
        img_size: Input image size
        patch_size: Patch size for ViT
        num_frames: Number of frames (must match checkpoint, typically 2 or 64)
        tubelet_size: Temporal patch size (must match checkpoint, typically 2)
    
    Returns:
        ViT-Giant model (1408-dim embeddings)
    
    Note: Even for static images, we use num_frames=2 and tubelet_size=2 to match
    the checkpoint architecture. We'll duplicate the image frame during inference.
    """
    from . import vision_transformer
    
    # ViT-Giant with RoPE (matches your e150.pt checkpoint)
    # Use tubelet_size=2 to match checkpoint weights
    model = vision_transformer.vit_giant_xformers_rope(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
    )
    
    return model


def create_vit_large(img_size=224, patch_size=16, num_frames=2, tubelet_size=2):
    """
    Create a ViT-Large model compatible with V-JEPA2 checkpoints.
    
    Args:
        img_size: Input image size
        patch_size: Patch size for ViT
        num_frames: Number of frames (must match checkpoint, typically 2 or 64)
        tubelet_size: Temporal patch size (must match checkpoint, typically 2)
    
    Returns:
        ViT-Large model (1024-dim embeddings)
    
    Note: Even for static images, we use num_frames=2 and tubelet_size=2 to match
    the checkpoint architecture. We'll duplicate the image frame during inference.
    """
    from . import vision_transformer
    
    # ViT-Large with RoPE
    # Use tubelet_size=2 to match checkpoint weights
    model = vision_transformer.vit_large_rope(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
    )
    
    return model


def load_vjepa2_weights(model, checkpoint_path, use_target_encoder=True):
    """
    Load V-JEPA2 weights into model.
    
    Args:
        model: ViT model to load weights into
        checkpoint_path: Path to checkpoint file
        use_target_encoder: If True, use 'target_encoder' (EMA), else use 'encoder'
    
    Returns:
        model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Get state dict
    if use_target_encoder and 'target_encoder' in checkpoint:
        state_dict = checkpoint['target_encoder']
        print("Loading from 'target_encoder' (EMA model)")
    elif 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
        print("Loading from 'encoder'")
    else:
        raise ValueError(f"No encoder found in checkpoint. Keys: {list(checkpoint.keys())}")
    
    # Clean keys (remove 'module.' or 'backbone.' prefixes if present)
    cleaned_state_dict = {}
    for key, val in state_dict.items():
        new_key = key.replace("module.", "").replace("backbone.", "")
        cleaned_state_dict[new_key] = val
    
    # Load state dict
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    
    print("Weights loaded successfully")
    return model

