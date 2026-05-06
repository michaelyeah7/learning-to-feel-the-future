"""
V-JEPA2 ViT Encoder with Patch-Level Residual Adapters

This module wraps the frozen V-JEPA2 ViT encoder and adds learnable residual adapters
that process all patch tokens (not just CLS token) for richer spatial feature learning.
"""

import torch
import torch.nn as nn
from ModelTrain.module.vitg_encoder import ViTGEncoderSimple
from ModelTrain.module.residual_adapter import PatchResidualAdapter, AttentionPooling, MeanPooling


class ViTGEncoderAdapter(nn.Module):
    """
    Adapter-enhanced V-JEPA2 ViT encoder.
    
    This wrapper adds learnable capacity to the frozen ViT encoder by:
    1. Extracting all patch tokens (not just CLS)
    2. Applying per-token residual adapter
    3. Aggregating with attention pooling
    
    The output shape matches the original ViTGEncoderSimple for drop-in compatibility.
    """
    
    def __init__(
        self,
        ckpt_path: str,
        adapter_hidden_dim: int = 512,
        adapter_depth: int = 3,
        adapter_dropout: float = 0.1,
        adapter_scale_init: float = 0.1,
        pooling_type: str = 'attention',
        input_size: int = 224,
        model_type: str = 'vitg',
    ):
        """
        Initialize adapter-enhanced ViT encoder.
        
        Args:
            ckpt_path: Path to V-JEPA2 checkpoint
            adapter_hidden_dim: Hidden dimension for adapter MLP
            adapter_depth: Number of adapter MLP layers
            adapter_dropout: Dropout probability
            adapter_scale_init: Initial residual scaling factor
            pooling_type: Type of pooling ('attention' or 'mean')
            input_size: Input image size
            model_type: ViT model type ('vitg' or 'vitl')
        """
        super().__init__()
        
        self.model_type = model_type
        self.pooling_type = pooling_type
        
        # Load frozen V-JEPA2 ViT encoder
        print(f"Loading frozen ViT-{model_type.upper()} encoder...")
        self.vitg_base = ViTGEncoderSimple(
            ckpt_path=ckpt_path,
            input_size=input_size,
            model_type=model_type,
        )
        
        # Get embedding dimension from base encoder
        self.embed_dim = self.vitg_base.embed_dim
        
        # Create patch-level residual adapter
        print(f"Creating patch-level residual adapter (hidden_dim={adapter_hidden_dim}, depth={adapter_depth})...")
        self.patch_adapter = PatchResidualAdapter(
            embed_dim=self.embed_dim,
            hidden_dim=adapter_hidden_dim,
            depth=adapter_depth,
            dropout=adapter_dropout,
            scale_init=adapter_scale_init,
        )
        
        # Create pooling layer
        if pooling_type == 'attention':
            print("Using attention-based pooling")
            self.pooling = AttentionPooling(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout=adapter_dropout,
            )
        elif pooling_type == 'mean':
            print("Using mean pooling")
            self.pooling = MeanPooling(embed_dim=self.embed_dim)
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}. Choose 'attention' or 'mean'")
        
        # Count parameters
        adapter_params = self.patch_adapter.get_num_params()
        pooling_params = self.pooling.get_num_params()
        total_params = adapter_params + pooling_params
        
        print(f"Adapter parameters: {adapter_params:,} ({adapter_params/1e6:.2f}M)")
        print(f"Pooling parameters: {pooling_params:,} ({pooling_params/1e6:.2f}M)")
        print(f"Total trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # Verify frozen ViT
        frozen_params = sum(1 for p in self.vitg_base.parameters() if not p.requires_grad)
        total_vit_params = sum(1 for p in self.vitg_base.parameters())
        print(f"ViT parameters: {total_vit_params} total, {frozen_params} frozen")
    
    def forward(self, x: torch.Tensor, draft_embedding: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with adapter and attention pooling.
        
        Args:
            x: Input tactile images (B, C, H, W)
            draft_embedding: Optional draft action embedding (B, embed_dim) to condition adapter
        
        Returns:
            Aggregated features (B, embed_dim) - same shape as ViTGEncoderSimple
        """
        # Get all patch tokens from frozen ViT (NOT just CLS)
        # The vitg_base.forward() is called within torch.no_grad() internally
        patches = self.vitg_base(x, return_all_tokens=True)  # (B, num_patches, embed_dim)
        
        # Optionally prepend draft embedding as first token (if provided)
        if draft_embedding is not None:
            draft_token = draft_embedding.unsqueeze(1)  # (B, 1, embed_dim)
            patches = torch.cat([draft_token, patches], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Apply residual adapter to all patches (including draft token if present)
        # Note: Adapter itself is trainable, so gradients flow here
        adapted_patches = self.patch_adapter(patches)  # (B, num_patches[+1], embed_dim)
        
        # Aggregate patches using attention pooling
        # If draft token is present, attention will learn to weight it vs tactile patches
        output = self.pooling(adapted_patches)  # (B, embed_dim)
        
        return output
    
    def get_num_params(self):
        """Return number of parameters in the encoder."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Return number of trainable parameters (adapter + pooling only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

