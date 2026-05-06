# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETRJEPAAdapter model - ACTJEPA with patch-level residual adapters on ViT encoders
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRJEPAAdapter(nn.Module):
    """ DETRJEPAAdapter: ACTJEPA with CLIP for RGB + patch-level residual adapters for tactile ViT """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, 
                 vq, vq_class, vq_dim, action_dim, vitg_ckpt_path, tactile_camera_names, 
                 vit_model='vitg', adapter_hidden_dim=512, adapter_depth=3, 
                 adapter_dropout=0.1, adapter_scale_init=0.1, adapter_pooling='attention', 
                 clip_encoder=None, text_embedding=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module list of ResNet backbones for RGB cameras (deprecated, use clip_encoder)
            transformer: torch module of the transformer architecture
            encoder: torch module of the VAE encoder
            state_dim: robot state dimension
            num_queries: number of action queries
            camera_names: list of RGB camera names
            vq, vq_class, vq_dim: vector quantization parameters
            action_dim: action dimension
            vitg_ckpt_path: path to V-JEPA2 ViT checkpoint file (.pt)
            tactile_camera_names: list of tactile sensor names
            vit_model: ViT model type ('vitg' or 'vitl')
            adapter_hidden_dim: hidden dimension for adapter MLP
            adapter_depth: number of adapter MLP layers
            adapter_dropout: dropout rate for adapter
            adapter_scale_init: initial residual scaling factor
            adapter_pooling: pooling type ('attention' or 'mean')
            clip_encoder: CLIPEncoder instance for RGB cameras (replaces ResNet backbones)
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names  # RGB cameras
        self.tactile_camera_names = tactile_camera_names
        self.all_camera_names = camera_names + tactile_camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # ACTJEPAAdapter uses hybrid architecture: CLIP for RGB + Adapter-enhanced ViT for tactile
        print(f"ACTJEPAAdapter: CLIP for {len(camera_names)} RGB cameras + Adapter-ViT-{vit_model.upper()} for {len(tactile_camera_names)} tactile sensors")
        
        # Use CLIP encoder for RGB cameras if provided, otherwise fallback to ResNet
        if clip_encoder is not None:
            self.clip_encoder = clip_encoder
            self.backbones = None
            self.use_clip = True
            print(f"  - Using CLIP encoder for RGB cameras")
        else:
            # Legacy ResNet mode
            self.backbones = nn.ModuleList(backbones)
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.use_clip = False
            print(f"  - Using ResNet backbones for RGB cameras (legacy mode)")
        
        # Create adapter-enhanced V-JEPA2 ViT encoder for tactile sensors (shared across all tactile sensors)
        from ModelTrain.module.vitg_encoder_adapter import ViTGEncoderAdapter
        print(f"Loading adapter-enhanced ViT-{vit_model.upper()} from: {vitg_ckpt_path}")
        # Load once and share across all tactile sensors to save memory
        # ViT processes all patch tokens through residual adapters before pooling
        self.vitg_encoder_shared = ViTGEncoderAdapter(
            ckpt_path=vitg_ckpt_path,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_depth=adapter_depth,
            adapter_dropout=adapter_dropout,
            adapter_scale_init=adapter_scale_init,
            pooling_type=adapter_pooling,
            input_size=224,
            model_type=vit_model,
        )
        
        # Get actual embedding dimension from encoder
        vit_embed_dim = self.vitg_encoder_shared.embed_dim
        
        # Project ViT embeddings to hidden_dim
        self.vitg_proj = nn.Linear(vit_embed_dim, hidden_dim)
        
        # Position embedding for tactile features
        self.tactile_pos_embed = nn.Parameter(torch.randn(1, hidden_dim, 1))
        
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        
        # Draft action projection for conditioning tactile encoder
        self.draft_action_proj = nn.Linear(action_dim, vit_embed_dim)
        print(f"Draft-then-refine enabled: draft action ({action_dim}) -> ViT embedding ({vit_embed_dim})")
        
        # Store text embedding if provided (for text-conditioned tasks)
        self.text_embedding = text_embedding
        if text_embedding is not None:
            print(f"Text conditioning enabled: text embedding shape {text_embedding.shape}")


    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward_draft(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        Draft forward pass: Process RGB through ResNet + MASKED (zero) tactile embeddings.
        
        This maintains the same sequence structure as full forward pass for transformer reusability:
        - RGB cameras: Normal ResNet processing
        - Tactile sensors: Zero embeddings (masked out)
        
        Args:
            qpos: batch, qpos_dim
            image: list of [rgb_data, tactile_data] or batch, num_cam, channel, height, width
            env_state: None
            actions: batch, seq, action_dim (for VAE encoder)
            is_pad: batch, seq (padding mask)
            vq_sample: VQ sample if using VQ
        
        Returns:
            a_hat_draft: Draft action predictions (batch, num_queries, action_dim)
            is_pad_hat: Padding predictions
            [mu, logvar]: Latent statistics
            probs, binaries: VQ outputs
        """
        # Encode latent z (same as full forward)
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)
        
        bs = qpos.shape[0]
        all_features = []
        all_pos = []
        
        # Extract RGB images only (tactile will be masked)
        if isinstance(image, list):
            rgb_images = image[0]  # (B, num_rgb, C, H, W)
            # Note: tactile_images = image[1] - will be MASKED (not used)
        else:
            rgb_images = image[:, :len(self.camera_names)]
        
        # Process RGB cameras
        if self.use_clip:
            # CLIP mode: Process RGB cameras through CLIP encoder
            for cam_id, cam_name in enumerate(self.camera_names):
                rgb_image = rgb_images[:, cam_id]
                # CLIP encoder returns (features, pos) where features are already projected
                features, pos = self.clip_encoder(rgb_image)
                all_features.append(features)
                all_pos.append(pos[:1])  # Use first batch item for pos (shared across batch)
        else:
            # Legacy ResNet mode
            for cam_id, cam_name in enumerate(self.camera_names):
                rgb_image = rgb_images[:, cam_id]
                features, pos = self.backbones[cam_id](rgb_image)
                features = features[0]  # (B, C, H, W)
                pos = pos[0]
                
                # Project and flatten to sequence
                projected = self.input_proj(features)  # (B, hidden_dim, H, W)
                projected = projected.flatten(2)  # (B, hidden_dim, H*W)
                all_features.append(projected)
                
                pos = pos.flatten(2)  # (1, hidden_dim, H*W)
                all_pos.append(pos)
        
        # Process tactile sensors with MASKED (zero) embeddings
        # This maintains sequence structure for transformer
        hidden_dim = self.transformer.d_model
        for tac_id, tac_name in enumerate(self.tactile_camera_names):
            # Create zero embedding (masked tactile feature)
            tac_feature = torch.zeros(bs, hidden_dim, 1, device=rgb_images.device)  # (B, hidden_dim, 1)
            all_features.append(tac_feature)
            
            # Position embedding (same as full forward)
            tac_pos = self.tactile_pos_embed  # (1, hidden_dim, 1)
            all_pos.append(tac_pos)
        
        # Concatenate all features (RGB + masked tactile)
        # Same sequence structure as full forward: [RGB tokens..., Tactile tokens...]
        src = torch.cat(all_features, dim=2)  # (B, hidden_dim, total_seq_len)
        pos = torch.cat(all_pos, dim=2)  # (1, hidden_dim, total_seq_len)
        
        # Reshape to 4D for transformer
        src = src.unsqueeze(2)  # (B, hidden_dim, 1, seq_len)
        pos = pos.unsqueeze(2)  # (1, hidden_dim, 1, seq_len)
        
        # Transformer decoder (reuses same architecture as full forward)
        proprio_input = self.input_proj_robot_state(qpos)
        hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        
        # Draft action prediction
        a_hat_draft = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        
        return a_hat_draft, is_pad_hat, [mu, logvar], probs, binaries

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None, use_draft=True):
        """
        Full forward pass with optional draft-then-refine strategy.
        
        Args:
            qpos: batch, qpos_dim
            image: list of [rgb_data, tactile_data] or batch, num_cam, channel, height, width
            env_state: None
            actions: batch, seq, action_dim
            is_pad: batch, seq (padding mask)
            vq_sample: VQ sample if using VQ
            use_draft: If True, first compute draft action (RGB + masked tactile), then condition tactile on it
        
        Returns:
            a_hat: Action predictions (batch, num_queries, action_dim)
            is_pad_hat: Padding predictions
            [mu, logvar]: Latent statistics
            probs, binaries: VQ outputs
        """
        # Optional: Generate draft action first (RGB + masked tactile, no gradients)
        if use_draft:
            with torch.no_grad():
                a_hat_draft, _, _, _, _ = self.forward_draft(qpos, image, env_state, actions, is_pad, vq_sample)
                # Use first timestep action as conditioning for tactile encoder
                draft_action = a_hat_draft[:, 0, :]  # (B, action_dim)
        else:
            draft_action = None
        
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # ACTJEPAAdapter uses hybrid mode: Process RGB through ResNet + Tactile through Adapter-ViT (conditioned on draft)
        bs = qpos.shape[0]
        all_features = []
        all_pos = []
        
        # Handle list input (for different resolutions) or tensor input
        if isinstance(image, list):
            rgb_images = image[0]  # (B, num_rgb, C, H, W)
            tactile_images = image[1]  # (B, num_tactile, C, H, W)
        else:
            # Single tensor input (same resolution for all)
            rgb_images = image[:, :len(self.camera_names)]
            tactile_images = image[:, len(self.camera_names):]
        
        # Process RGB cameras
        if self.use_clip:
            # CLIP mode: Process RGB cameras through CLIP encoder
            for cam_id, cam_name in enumerate(self.camera_names):
                rgb_image = rgb_images[:, cam_id]
                # CLIP encoder returns (features, pos) where features are already projected
                # Shape: features (B, hidden_dim, num_patches), pos (B, hidden_dim, num_patches)
                features, pos = self.clip_encoder(rgb_image)
                all_features.append(features)
                all_pos.append(pos[:1])  # Use first batch item for pos (shared across batch)
        else:
            # Legacy ResNet mode
            for cam_id, cam_name in enumerate(self.camera_names):
                rgb_image = rgb_images[:, cam_id]
                features, pos = self.backbones[cam_id](rgb_image)
                features = features[0]  # take the last layer feature (B, C, H, W)
                pos = pos[0]
                
                # Project and flatten to sequence
                projected = self.input_proj(features)  # (B, hidden_dim, H, W)
                projected = projected.flatten(2)  # (B, hidden_dim, H*W) - flatten spatial
                all_features.append(projected)
                
                # Flatten position embeddings
                pos = pos.flatten(2)  # (1, hidden_dim, H*W)
                all_pos.append(pos)
        
        # Process tactile sensors through Adapter-ViT (shared encoder)
        # The adapter processes all patch tokens and pools them
        # Optionally conditioned on draft action if use_draft=True
        for tac_id, tac_name in enumerate(self.tactile_camera_names):
            tactile_image = tactile_images[:, tac_id]
            
            # Project draft action to ViT embedding space if available
            if draft_action is not None:
                draft_embedding = self.draft_action_proj(draft_action)  # (B, vit_embed_dim)
            else:
                draft_embedding = None
            
            # Get adapter-enhanced ViT embedding (conditioned on draft if provided)
            # Passes draft_embedding to adapter for cross-attention/conditioning
            tac_embedding = self.vitg_encoder_shared(tactile_image, draft_embedding=draft_embedding)  # (B, embed_dim)
            
            # Project to hidden_dim
            tac_feature = self.vitg_proj(tac_embedding)  # (B, hidden_dim)
            
            # Reshape to sequence format: (B, hidden_dim, 1) to match flattened RGB
            tac_feature = tac_feature.unsqueeze(-1)  # (B, hidden_dim, 1) - single token
            all_features.append(tac_feature)
            
            # Position embedding as sequence: (1, hidden_dim, 1)
            tac_pos = self.tactile_pos_embed  # (1, hidden_dim, 1)
            all_pos.append(tac_pos)
        
        # Concatenate all features along sequence dimension
        # RGB: (B, hidden_dim, H*W) per camera
        # Tactile: (B, hidden_dim, 1) per sensor
        # Total: unified sequence of all visual tokens
        src = torch.cat(all_features, dim=2)  # (B, hidden_dim, total_seq_len)
        pos = torch.cat(all_pos, dim=2)  # (1, hidden_dim, total_seq_len)
        
        # Reshape to 4D for transformer (it expects (B, C, H, W) format)
        # Treat sequence as width dimension: (B, hidden_dim, 1, seq_len)
        src = src.unsqueeze(2)  # (B, hidden_dim, 1, seq_len)
        pos = pos.unsqueeze(2)  # (1, hidden_dim, 1, seq_len)
        
        # Proprioception and transformer
        proprio_input = self.input_proj_robot_state(qpos)
        
        # Add text embedding to latent if available (text conditioning)
        if self.text_embedding is not None:
            # Expand text embedding to batch size and add to latent
            text_emb_expanded = self.text_embedding.unsqueeze(0).expand(bs, -1).to(latent_input.device)
            # Concatenate or add text embedding to latent (you can modify this)
            latent_input = latent_input + text_emb_expanded
        
        hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build_jepa_adapter(args):
    """Build ACTJEPAAdapter model with hybrid CLIP + Adapter-enhanced ViT architecture"""
    state_dim = 14 # TODO hardcode

    # ACTJEPAAdapter requires both RGB and tactile cameras
    vitg_ckpt_path = getattr(args, 'vitg_ckpt_path', None) or getattr(args, 'vit_ckpt_path', None)
    tactile_camera_names = getattr(args, 'tactile_camera_names', [])
    
    if not vitg_ckpt_path:
        raise ValueError("ACTJEPAAdapter requires vitg_ckpt_path or vit_ckpt_path to be specified")
    if not tactile_camera_names:
        raise ValueError("ACTJEPAAdapter requires tactile_camera_names to be specified")
    
    # Create CLIP encoder for RGB cameras if specified, otherwise use ResNet
    clip_encoder = None
    backbones = None
    
    if hasattr(args, 'clip_model') and args.clip_model:
        # Create CLIP encoder for RGB cameras
        from ModelTrain.module.clip_encoder import create_clip_encoder
        print(f"Creating CLIP encoder: {args.clip_model}")
        clip_encoder = create_clip_encoder(
            model_name=args.clip_model,
            pretrained=getattr(args, 'clip_pretrained', 'openai'),
            hidden_dim=args.hidden_dim,
            freeze=getattr(args, 'freeze_clip', False),
            image_size=224,  # CLIP default
            enable_text=getattr(args, 'enable_text', False)
        )
    else:
        # Legacy: Build ResNet backbones for RGB cameras only (not tactile)
        backbones = []
        for cam_name in args.camera_names:
            if cam_name not in tactile_camera_names:
                backbone = build_backbone(args)
                backbones.append(backbone)
        
        if len(backbones) == 0:
            raise ValueError("ACTJEPAAdapter requires at least one RGB camera")
    
    # Get ViT model type and adapter config from args
    vit_model = getattr(args, 'vit_model', 'vitg')
    adapter_hidden_dim = getattr(args, 'adapter_hidden_dim', 512)
    adapter_depth = getattr(args, 'adapter_depth', 3)
    adapter_dropout = getattr(args, 'adapter_dropout', 0.1)
    adapter_scale_init = getattr(args, 'adapter_scale_init', 0.1)
    adapter_pooling = getattr(args, 'adapter_pooling', 'attention')
    
    if clip_encoder:
        print(f"Building ACTJEPAAdapter: CLIP encoder + {len(tactile_camera_names)} Adapter-ViT-{vit_model.upper()} encoders")
    else:
        print(f"Building ACTJEPAAdapter: {len(backbones)} ResNet backbones + {len(tactile_camera_names)} Adapter-ViT-{vit_model.upper()} encoders")
    print(f"Adapter config: hidden_dim={adapter_hidden_dim}, depth={adapter_depth}, dropout={adapter_dropout}, scale={adapter_scale_init}, pooling={adapter_pooling}")

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    # Encode text prompt if provided
    text_embedding = None
    if clip_encoder is not None and hasattr(clip_encoder, 'enable_text') and clip_encoder.enable_text:
        text_prompt = getattr(args, 'text_prompt', None)
        if text_prompt:
            print(f"Encoding text prompt: '{text_prompt}'")
            with torch.no_grad():  # Text encoding doesn't need gradients during training
                text_embedding = clip_encoder.encode_text(text_prompt).squeeze(0)  # (hidden_dim,)
            print(f"Text embedding shape: {text_embedding.shape}")
    
    model = DETRJEPAAdapter(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
        vitg_ckpt_path=vitg_ckpt_path,
        tactile_camera_names=tactile_camera_names,
        vit_model=vit_model,
        adapter_hidden_dim=adapter_hidden_dim,
        adapter_depth=adapter_depth,
        adapter_dropout=adapter_dropout,
        adapter_scale_init=adapter_scale_init,
        adapter_pooling=adapter_pooling,
        clip_encoder=clip_encoder,
        text_embedding=text_embedding,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameters: %.2fM" % (n_parameters/1e6,))

    return model

