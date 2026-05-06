# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETRJEPA model - Hybrid architecture with ResNet for RGB + V-JEPA2 ViTG for tactile
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


class DETRJEPA(nn.Module):
    """ DETRJEPA: Hybrid ACT model with CLIP for RGB + V-JEPA2 ViT for tactile """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim, vitg_ckpt_path, tactile_camera_names, vit_model='vitg', clip_encoder=None):
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
        
        # ACTJEPA uses hybrid architecture: CLIP for RGB + ViT for tactile
        print(f"ACTJEPA: CLIP for {len(camera_names)} RGB cameras + ViT-{vit_model.upper()} for {len(tactile_camera_names)} tactile sensors")
        
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
        
        # Create V-JEPA2 ViT encoder for tactile sensors (shared across all tactile sensors)
        from ModelTrain.module.vitg_encoder import ViTGEncoderSimple
        print(f"Loading V-JEPA2 ViT-{vit_model.upper()} from: {vitg_ckpt_path}")
        # Load once and share across all tactile sensors to save memory
        # ViT-Giant outputs 1408-dim embeddings, ViT-Large outputs 1024-dim embeddings
        self.vitg_encoder_shared = ViTGEncoderSimple(vitg_ckpt_path, input_size=224, model_type=vit_model)
        
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

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        qpos: batch, qpos_dim
        image: list of [rgb_data, tactile_data] or batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # ACTJEPA hybrid mode: Process RGB through CLIP (or ResNet legacy) + Tactile through ViTG
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
        
        # Process tactile sensors through ViTG (shared encoder)
        for tac_id, tac_name in enumerate(self.tactile_camera_names):
            tactile_image = tactile_images[:, tac_id]
            
            # Get ViTG embedding using shared encoder
            tac_embedding = self.vitg_encoder_shared(tactile_image)  # (B, 1408)
            
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


def build_jepa(args):
    """Build ACTJEPA model with hybrid CLIP+ViTG architecture"""
    state_dim = 14 # TODO hardcode

    # ACTJEPA always requires both RGB and tactile cameras
    vitg_ckpt_path = getattr(args, 'vitg_ckpt_path', None)
    tactile_camera_names = getattr(args, 'tactile_camera_names', [])
    
    if not vitg_ckpt_path:
        raise ValueError("ACTJEPA requires vitg_ckpt_path to be specified")
    if not tactile_camera_names:
        raise ValueError("ACTJEPA requires tactile_camera_names to be specified")
    
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
            raise ValueError("ACTJEPA requires at least one RGB camera")
    
    # Get ViT model type from args
    vit_model = getattr(args, 'vit_model', 'vitg')
    
    if clip_encoder:
        print(f"Building ACTJEPA: CLIP encoder + {len(tactile_camera_names)} ViT-{vit_model.upper()} encoders")
    else:
        print(f"Building ACTJEPA: {len(backbones)} ResNet backbones + {len(tactile_camera_names)} ViT-{vit_model.upper()} encoders")

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    model = DETRJEPA(
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
        clip_encoder=clip_encoder,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

