"""
CLIP Vision Encoder Wrapper for RGB Image Processing

This module provides a wrapper around OpenAI's CLIP vision encoder
to process RGB camera images and produce features for the ACT policy.

Key features:
- Loads pretrained CLIP weights (ViT-B/16, ViT-L/14, etc.)
- Extracts patch tokens (not CLS token) for spatial features
- Projects to policy's hidden_dim
- Supports frozen and trainable modes
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional, Tuple

# Try to import open_clip
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("WARNING: open_clip not available. Install with: pip install open_clip_torch")


class CLIPEncoder(nn.Module):
    """
    CLIP vision encoder wrapper for RGB image processing.
    
    Uses pretrained CLIP models and extracts spatial patch tokens
    for use in transformer-based policies.
    Also optionally includes text encoding capability.
    """
    
    def __init__(self,
                 model_name: str = 'ViT-B-16',
                 pretrained: str = 'openai',
                 hidden_dim: int = 512,
                 freeze: bool = False,
                 image_size: int = 224,
                 enable_text: bool = False):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: CLIP model architecture ('ViT-B-16', 'ViT-B-32', 'ViT-L-14', etc.)
            pretrained: Pretrained weights to use ('openai', 'laion2b_s34b_b88k', etc.)
            hidden_dim: Output feature dimension (for projection layer)
            freeze: If True, freeze CLIP weights (no gradient updates)
            image_size: Input image size (CLIP default is 224)
            enable_text: If True, also initialize text encoder for language conditioning
        """
        super().__init__()
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip is required. Install with: pip install open_clip_torch")
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.freeze = freeze
        self.image_size = image_size
        self.enable_text = enable_text
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name} with {pretrained} weights")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            image_size=image_size
        )
        
        # Load tokenizer if text encoding is enabled
        if enable_text:
            self.tokenizer = open_clip.get_tokenizer(model_name)
            print(f"CLIP text encoder enabled")
        
        # Get CLIP feature dimension
        self.clip_dim = self.clip_model.visual.output_dim
        
        # Calculate number of patches
        if hasattr(self.clip_model.visual, 'patch_size'):
            patch_size = self.clip_model.visual.patch_size[0] if isinstance(
                self.clip_model.visual.patch_size, tuple) else self.clip_model.visual.patch_size
        else:
            # Default for ViT models
            patch_size = 16 if 'B-16' in model_name or 'L-14' in model_name else 32
        
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # Projection layer: CLIP features -> policy hidden_dim
        self.projection = nn.Linear(self.clip_dim, hidden_dim)
        
        # Position embeddings for transformer input
        # Shape: (1, hidden_dim, num_patches)
        self.pos_embed = nn.Parameter(torch.randn(1, hidden_dim, self.num_patches))
        
        # Text projection if text encoding is enabled
        if enable_text:
            self.text_projection = nn.Linear(self.clip_dim, hidden_dim)
        
        # Freeze CLIP weights if requested
        if freeze:
            self._freeze_clip()
        
        print(f"CLIP Encoder initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - CLIP dim: {self.clip_dim}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Patch size: {patch_size}")
        print(f"  - Num patches: {self.num_patches} ({self.num_patches_per_side}x{self.num_patches_per_side})")
        print(f"  - Frozen: {freeze}")
        print(f"  - Text encoding: {enable_text}")
    
    def _freeze_clip(self):
        """Freeze CLIP model parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print("CLIP encoder frozen (all CLIP parameters set to requires_grad=False)")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images through CLIP visual encoder.
        
        Args:
            images: Input images, shape (B, C, H, W)
        
        Returns:
            Patch tokens, shape (B, num_patches, clip_dim)
        """
        # Use CLIP's visual encoder
        x = images
        
        # Get visual features from CLIP
        # Most CLIP models have a visual attribute
        visual = self.clip_model.visual
        
        # Process through CLIP ViT
        x = visual.conv1(x)  # Patch embedding
        
        # Reshape to sequence: (B, clip_dim, grid_h, grid_w) -> (B, clip_dim, num_patches) -> (B, num_patches, clip_dim)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # (B, num_patches, clip_dim)
        
        # Add class token and positional embedding
        x = torch.cat([
            visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        
        # Apply pre-norm
        x = visual.ln_pre(x)
        
        # Transformer blocks
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim) for transformer
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, dim)
        
        # Remove CLS token, keep only patch tokens
        patch_tokens = x[:, 1:, :]  # (B, num_patches, clip_dim)
        
        return patch_tokens
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CLIP encoder.
        
        Args:
            images: Input images, shape (B, C, H, W)
                   Images should be normalized with CLIP's normalization
        
        Returns:
            features: Projected features, shape (B, hidden_dim, num_patches)
            pos: Position embeddings, shape (1, hidden_dim, num_patches)
        """
        # Get patch tokens from CLIP
        if self.freeze:
            with torch.no_grad():
                patch_tokens = self.encode_image(images)  # (B, num_patches, clip_dim)
        else:
            patch_tokens = self.encode_image(images)  # (B, num_patches, clip_dim)
        
        # Project to hidden_dim
        projected = self.projection(patch_tokens)  # (B, num_patches, hidden_dim)
        
        # Transpose to match ResNet output format: (B, hidden_dim, num_patches)
        features = projected.permute(0, 2, 1)  # (B, hidden_dim, num_patches)
        
        # Position embeddings
        pos = self.pos_embed.expand(images.shape[0], -1, -1)  # (B, hidden_dim, num_patches)
        
        return features, pos
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for CLIP encoder.
        
        Args:
            images: Raw images, shape (B, C, H, W), values in [0, 255] or [0, 1]
        
        Returns:
            Preprocessed images ready for CLIP
        """
        # Normalize to [0, 1] if needed
        if images.max() > 1.0:
            images = images / 255.0
        
        # CLIP normalization
        # Mean and std from CLIP preprocessing
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        
        images = (images - mean) / std
        
        return images
    
    def encode_text(self, text_prompts):
        """
        Encode text prompts using CLIP text encoder.
        
        Args:
            text_prompts: List of text strings or single text string
        
        Returns:
            Text embeddings, shape (B, hidden_dim) where B is number of prompts
        """
        if not self.enable_text:
            raise RuntimeError("Text encoding not enabled. Set enable_text=True during initialization.")
        
        # Convert to list if single string
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # Tokenize text
        text_tokens = self.tokenizer(text_prompts).to(next(self.parameters()).device)
        
        # Encode text
        if self.freeze:
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
        else:
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Project to hidden_dim
        text_embeddings = self.text_projection(text_features)  # (B, hidden_dim)
        
        return text_embeddings
    
    def get_num_params(self) -> int:
        """Return the number of parameters in the encoder."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CLIP Encoder: {total:,} total parameters, {trainable:,} trainable")
        return total


def create_clip_encoder(model_name: str = 'ViT-B-16',
                        pretrained: str = 'openai',
                        hidden_dim: int = 512,
                        freeze: bool = False,
                        image_size: int = 224,
                        enable_text: bool = False) -> CLIPEncoder:
    """
    Factory function to create a CLIP encoder.
    
    Args:
        model_name: CLIP model architecture
        pretrained: Pretrained weights
        hidden_dim: Output feature dimension
        freeze: Whether to freeze CLIP weights
        image_size: Input image size
        enable_text: Whether to enable text encoding
    
    Returns:
        CLIPEncoder instance
    """
    return CLIPEncoder(
        model_name=model_name,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        freeze=freeze,
        image_size=image_size,
        enable_text=enable_text
    )

