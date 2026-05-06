"""
Residual Adapter Modules for V-JEPA2 ViT Encoders

This module provides learnable adapter networks that can be applied to frozen
ViT encoders to enable task-specific learning while preserving pre-trained knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchResidualAdapter(nn.Module):
    """
    Residual adapter that processes patch token sequences.
    
    Applies a per-token MLP with residual connections and learnable scaling.
    The adapter learns task-specific features while preserving frozen ViT representations.
    
    Architecture:
        input (B, N, embed_dim) 
        → LayerNorm 
        → MLP (embed_dim → hidden_dim → ... → embed_dim)
        → Dropout
        → Residual: output = input + scale * adapted
    """
    
    def __init__(
        self,
        embed_dim: int = 1408,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.1,
        scale_init: float = 0.1,
    ):
        """
        Initialize patch-level residual adapter.
        
        Args:
            embed_dim: Dimension of ViT embeddings (1408 for ViT-G, 1024 for ViT-L)
            hidden_dim: Hidden dimension for adapter MLP
            depth: Number of MLP layers (minimum 2)
            dropout: Dropout probability
            scale_init: Initial value for residual scaling factor
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.depth = max(2, depth)  # Ensure at least 2 layers
        
        # Layer normalization before adapter
        self.norm = nn.LayerNorm(embed_dim)
        
        # Build MLP layers
        layers = []
        
        # First layer: embed_dim -> hidden_dim
        layers.extend([
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(self.depth - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        # Last layer: hidden_dim -> embed_dim
        layers.append(nn.Linear(hidden_dim, embed_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Learnable residual scaling factor
        # Initialized small to prevent disrupting frozen features early in training
        self.scale = nn.Parameter(torch.tensor(scale_init))
        
        # Dropout after MLP
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights with small values for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights with small values for training stability."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Small initialization for residual stability
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input patch tokens (B, num_patches, embed_dim)
        
        Returns:
            Adapted tokens with residual: (B, num_patches, embed_dim)
        """
        # x: (B, N, embed_dim)
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Apply MLP (operates on last dimension, preserves patch dimension)
        x = self.mlp(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Residual connection with learnable scaling
        output = residual + self.scale * x
        
        return output
    
    def get_num_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate patch tokens into a single vector.
    
    Uses a learnable query to compute attention weights over all patch tokens,
    then returns a weighted sum. This is more flexible than just taking the CLS token.
    
    Architecture:
        - Learnable query vector
        - Multi-head attention: query attends to patch tokens
        - Output: weighted aggregation of patches
    """
    
    def __init__(
        self,
        embed_dim: int = 1408,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize attention pooling.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Learnable query token for pooling
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize query
        nn.init.normal_(self.query, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate patch tokens using attention.
        
        Args:
            x: Patch tokens (B, num_patches, embed_dim)
        
        Returns:
            Aggregated vector (B, embed_dim)
        """
        B, N, D = x.shape
        
        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # (B, 1, embed_dim)
        
        # Apply multi-head attention
        # query attends to patch tokens (keys and values)
        output, attn_weights = self.attention(
            query=query,      # (B, 1, embed_dim)
            key=x,           # (B, N, embed_dim)
            value=x,         # (B, N, embed_dim)
        )
        
        # output: (B, 1, embed_dim)
        output = output.squeeze(1)  # (B, embed_dim)
        
        # Layer norm
        output = self.norm(output)
        
        return output
    
    def get_num_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MeanPooling(nn.Module):
    """
    Simple mean pooling as an alternative to attention pooling.
    Just averages all patch tokens along the sequence dimension.
    """
    
    def __init__(self, embed_dim: int = 1408):
        """
        Initialize mean pooling.
        
        Args:
            embed_dim: Dimension of embeddings (for interface compatibility)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Average patch tokens.
        
        Args:
            x: Patch tokens (B, num_patches, embed_dim)
        
        Returns:
            Averaged vector (B, embed_dim)
        """
        # Simple mean over sequence dimension
        output = x.mean(dim=1)  # (B, embed_dim)
        output = self.norm(output)
        return output
    
    def get_num_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

