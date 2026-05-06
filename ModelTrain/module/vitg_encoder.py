"""
V-JEPA2 ViTG Encoder Wrapper for Tactile Image Processing

This module provides a frozen ViTG encoder that processes tactile images
and produces 1280-dimensional embeddings for the ACT policy.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional

# Try to import V-JEPA2 model architecture (local copy for Python 3.8 compatibility)
try:
    # Use local vjepa2_compat directory (Python 3.8 compatible)
    from ModelTrain.vjepa2_compat.model_builder import create_vit_giant, create_vit_large, load_vjepa2_weights
    VJEPA_AVAILABLE = True
except ImportError as e:
    VJEPA_AVAILABLE = False
    print(f"WARNING: V-JEPA2 models not available: {e}")


class ViTGEncoder(nn.Module):
    """
    Wrapper for V-JEPA2 ViTG encoder to process tactile images.
    
    The encoder is frozen (all parameters have requires_grad=False) and produces
    1280-dimensional embeddings from input tactile images.
    """
    
    def __init__(self, ckpt_path: str, input_size: int = 224):
        """
        Initialize ViTG encoder from checkpoint.
        
        Args:
            ckpt_path: Path to the V-JEPA2 ViTG checkpoint (.pt file)
            input_size: Expected input image size (default: 224)
        """
        super().__init__()
        
        self.input_size = input_size
        self.embed_dim = 1280  # ViT-G standard embedding dimension
        
        # Load the checkpoint
        print(f"Loading ViTG checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Extract the model from checkpoint
        # The checkpoint structure may vary, so we need to handle different formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
            elif 'encoder' in checkpoint:
                # V-JEPA may store encoder separately
                model_state = checkpoint['encoder']
            else:
                # Assume the checkpoint itself is the state dict
                model_state = checkpoint
        else:
            # If checkpoint is directly a model
            self.encoder = checkpoint
            model_state = None
        
        # If we have a state dict, we need to create the model architecture
        # For V-JEPA2 ViTG, we'll try to use the model directly if available
        if model_state is not None:
            # Try to infer model architecture from state dict keys
            # V-JEPA uses a vision transformer architecture
            try:
                # Attempt to create a compatible ViT-G architecture
                self.encoder = self._create_vitg_model()
                # Load the state dict, being permissive about mismatches
                missing_keys, unexpected_keys = self.encoder.load_state_dict(model_state, strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
            except Exception as e:
                print(f"Error loading state dict: {e}")
                print("Attempting to use checkpoint directly as model...")
                self.encoder = checkpoint
        
        # Freeze all parameters
        self._freeze_encoder()
        
        # Set to eval mode
        self.encoder.eval()
        
        # Define preprocessing transforms
        # V-JEPA typically uses ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"ViTG encoder loaded successfully. Embedding dim: {self.embed_dim}")
    
    def _create_vitg_model(self):
        """
        Create a ViT-G model architecture.
        This is a placeholder - actual architecture depends on V-JEPA2 implementation.
        """
        # This would need to match the exact V-JEPA2 architecture
        # For now, we return a dummy module that will be replaced
        # In practice, you'd import the actual V-JEPA2 model class
        raise NotImplementedError(
            "Please ensure the checkpoint contains the full model, "
            "or import the V-JEPA2 model architecture explicitly."
        )
    
    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("ViTG encoder frozen (all parameters set to requires_grad=False)")
    
    def forward(self, x: torch.Tensor, return_cls_only: bool = True) -> torch.Tensor:
        """
        Forward pass through ViTG encoder.
        
        Args:
            x: Input tactile images, shape (B, C, H, W)
            return_cls_only: If True, return only CLS token embedding (B, 1280)
                           If False, return all patch embeddings (B, N, 1280)
        
        Returns:
            embeddings: Tensor of shape (B, 1280) if return_cls_only=True,
                       otherwise (B, N, 1280) where N is number of patches
        """
        # Preprocess images
        x = self.preprocess(x)
        
        # Pass through encoder (no gradient computation)
        with torch.no_grad():
            # V-JEPA encoders typically return a tuple or dict
            output = self.encoder(x)
            
            # Handle different output formats
            if isinstance(output, tuple):
                # Usually (features, intermediates) or similar
                features = output[0]
            elif isinstance(output, dict):
                # May have 'cls_token', 'patch_tokens', etc.
                if 'cls_token' in output:
                    features = output['cls_token']
                elif 'last_hidden_state' in output:
                    features = output['last_hidden_state']
                else:
                    # Take the first value
                    features = list(output.values())[0]
            else:
                features = output
            
            # Extract CLS token if needed
            if return_cls_only:
                if features.dim() == 3:  # (B, N, D)
                    # First token is typically CLS
                    features = features[:, 0, :]
                elif features.dim() == 2:  # (B, D)
                    # Already extracted
                    pass
                else:
                    raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        return features
    
    def get_num_params(self):
        """Return the number of parameters in the encoder."""
        return sum(p.numel() for p in self.encoder.parameters())


class ViTGEncoderSimple(nn.Module):
    """
    V-JEPA2 ViT encoder wrapper for tactile image processing.
    Loads V-JEPA2 checkpoint and creates frozen encoder.
    Supports both ViT-Giant (1408-dim) and ViT-Large (1024-dim).
    """
    
    def __init__(self, ckpt_path: str, embed_dim: int = None, input_size: int = 224, model_type: str = 'vitg'):
        super().__init__()
        
        self.model_type = model_type
        self.input_size = input_size
        
        # Set embed_dim based on model_type if not explicitly provided
        if embed_dim is None:
            if model_type == 'vitg':
                self.embed_dim = 1408
            elif model_type == 'vitl':
                self.embed_dim = 1024
            else:
                raise ValueError(f"Unknown model_type: {model_type}. Choose 'vitg' or 'vitl'")
        else:
            self.embed_dim = embed_dim
        
        print(f"Loading ViT-{model_type.upper()} checkpoint from: {ckpt_path}")
        
        # Load checkpoint directly to GPU to save RAM
        checkpoint = torch.load(ckpt_path, map_location='cuda')
        
        # Handle different checkpoint formats
        if hasattr(checkpoint, 'eval'):
            # Checkpoint is already a model
            self.encoder = checkpoint
            print("Loaded full model from checkpoint")
        elif isinstance(checkpoint, dict):
            # Checkpoint is a dictionary with state_dicts
            # print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # V-JEPA2 checkpoints contain state_dicts, need to instantiate model
            if not VJEPA_AVAILABLE:
                raise ImportError(
                    "V-JEPA2 model architecture not available.\n"
                    "The local V-JEPA2 model files should be in ModelTrain/vjepa2_compat/\n"
                    "Check that backbones.py and vision_transformer.py exist there."
                )
            
            # Create V-JEPA2 ViT model based on model_type
            # Note: Must match checkpoint architecture (tubelet_size=2 for video models)
            print(f"Creating V-JEPA2 ViT-{model_type.upper()} model (img_size={input_size}, tubelet_size=2)")
            if model_type == 'vitg':
                self.encoder = create_vit_giant(
                    img_size=input_size,
                    patch_size=16,
                    num_frames=2,  # Match checkpoint (will duplicate frames for static images)
                    tubelet_size=2,  # Match checkpoint architecture
                )
            elif model_type == 'vitl':
                self.encoder = create_vit_large(
                    img_size=input_size,
                    patch_size=16,
                    num_frames=2,  # Match checkpoint (will duplicate frames for static images)
                    tubelet_size=2,  # Match checkpoint architecture
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}. Choose 'vitg' or 'vitl'")
            
            # Load weights using helper function
            use_target = 'target_encoder' in checkpoint
            self.encoder = load_vjepa2_weights(self.encoder, ckpt_path, use_target_encoder=use_target)
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
        
        # Move to GPU and freeze encoder
        self.encoder.cuda()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.eval()
        
        print(f"ViT-{model_type.upper()} encoder loaded and frozen. Embed dim: {self.embed_dim}")
    
    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        """
        Forward pass returning CLS token embeddings or all patch tokens.
        
        Args:
            x: Input images (B, C, H, W), assumed to be already resized and normalized
            return_all_tokens: If True, return all patch tokens (B, num_patches, embed_dim)
                             If False, return only CLS token (B, embed_dim)
        
        Returns:
            CLS embeddings (B, embed_dim) if return_all_tokens=False
            All patch tokens (B, num_patches, embed_dim) if return_all_tokens=True
        """
        # V-JEPA2 expects images to be already normalized (done in dataset)
        # No additional preprocessing needed here
        
        # Forward pass without gradients
        with torch.no_grad():
            # V-JEPA2 models expect (B, C, num_frames, H, W) for videos
            # For static images, duplicate the frame to match num_frames=2
            if x.dim() == 4:  # (B, C, H, W)
                x = x.unsqueeze(2)  # (B, C, 1, H, W)
                # Duplicate frame to match model's expected num_frames=2
                x = x.repeat(1, 1, 2, 1, 1)  # (B, C, 2, H, W)
            
            # Forward through V-JEPA2 encoder
            features = self.encoder(x)
            
            # V-JEPA2 outputs patch tokens: (B, num_patches, embed_dim)
            # Extract CLS token or return all tokens based on flag
            if isinstance(features, (tuple, list)):
                features = features[0]
            
            if features.dim() == 3:  # (B, num_patches, embed_dim)
                if return_all_tokens:
                    # Return all patch tokens
                    return features  # (B, num_patches, embed_dim)
                else:
                    # Use CLS token (first token)
                    features = features[:, 0, :]  # Take CLS token
            elif features.dim() == 2:  # (B, embed_dim)
                # Already extracted (shouldn't happen with return_all_tokens=True)
                pass
        
        return features

