# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_ACTJEPA_model, build_CNNMLP_model

import IPython
e = IPython.embed

# def get_args_parser():
#     parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
#     parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
#     parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
#     parser.add_argument('--batch_size', default=2, type=int) # not used
#     parser.add_argument('--weight_decay', default=1e-4, type=float)
#     parser.add_argument('--epochs', default=300, type=int) # not used
#     parser.add_argument('--lr_drop', default=200, type=int) # not used
#     parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
#                         help='gradient clipping max norm')
#
#     # Model parameters
#     # * Backbone
#     parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
#                         help="Name of the convolutional backbone to use")
#     parser.add_argument('--dilation', action='store_true',
#                         help="If true, we replace stride with dilation in the last convolutional block (DC5)")
#     parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
#                         help="Type of positional embedding to use on top of the image features")
#     parser.add_argument('--camera_names', default=[], type=list, # will be overridden
#                         help="A list of camera names")
#
#     # * Transformer
#     parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
#                         help="Number of encoding layers in the transformer")
#     parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
#                         help="Number of decoding layers in the transformer")
#     parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
#                         help="Intermediate size of the feedforward layers in the transformer blocks")
#     parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
#                         help="Size of the embeddings (dimension of the transformer)")
#     parser.add_argument('--dropout', default=0.1, type=float,
#                         help="Dropout applied in the transformer")
#     parser.add_argument('--nheads', default=8, type=int, # will be overridden
#                         help="Number of attention heads inside the transformer's attentions")
#     parser.add_argument('--num_queries', default=400, type=int, # will be overridden
#                         help="Number of query slots")
#     parser.add_argument('--pre_norm', action='store_true')
#
#     # * Segmentation
#     parser.add_argument('--masks', action='store_true',
#                         help="Train segmentation head if the flag is provided")
#
#     # repeat args in imitate_episodes just to avoid error. Will not be used
#     parser.add_argument('--eval', action='store_true')
#     parser.add_argument('--onscreen_render', action='store_true')
#     parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir',required=True)
#     parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
#     parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
#     parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
#     parser.add_argument('--num_steps', action='store', type=int, help='num_epochs', required=True)
#     parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
#     parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
#     parser.add_argument('--temporal_agg', action='store_true')
#
#     parser.add_argument('--use_vq', action='store_true')
#     parser.add_argument('--vq_class', action='store', type=int, help='vq_class', required=False)
#     parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim', required=False)
#     parser.add_argument('--load_pretrain', action='store_true', default=False)
#     parser.add_argument('--action_dim', action='store', type=int, required=False)
#     parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
#     parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
#     parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
#     parser.add_argument('--resume_ckpt_path', action='store', type=str, help='load_ckpt_path', required=False)
#     parser.add_argument('--no_encoder', action='store_true')
#     parser.add_argument('--skip_mirrored_data', action='store_true')
#     parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
#     parser.add_argument('--history_len', action='store', type=int)
#     parser.add_argument('--future_len', action='store', type=int)
#     parser.add_argument('--prediction_len', action='store', type=int)
#
#     return parser

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)  # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # will be overridden
    parser.add_argument('--batch_size', default=2, type=int)  # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)  # not used
    parser.add_argument('--lr_drop', default=200, type=int)  # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float,  # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,  # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list,  # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,  # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,  # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,  # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,  # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,  # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int,  # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default='../ckpt', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT',required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='sim_transfer_cube_scripted', required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0,required=False)
    parser.add_argument('--num_steps', action='store', type=int, help='num_epochs', default=2000,required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class', required=False)
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim', required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--action_dim', action='store', type=int, required=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every',
                        required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='load_ckpt_path', required=False)
    parser.add_argument('--no_encoder', action='store_true')
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)
    
    # ViT-related arguments (for ACTJEPA)
    parser.add_argument('--use_vitg', action='store_true', default=False, help='Use ViTG encoder for tactile images (deprecated)')
    parser.add_argument('--vit_model', action='store', type=str, default='vitg', 
                        choices=['vitg', 'vitl'],
                        help='ViT model type for tactile processing: vitg (1408-dim) or vitl (1024-dim)')
    parser.add_argument('--vit_ckpt_path', action='store', type=str, help='Path to ViT checkpoint file (.pt)', required=False)
    parser.add_argument('--vitg_ckpt_path', action='store', type=str, help='Path to ViTG checkpoint file (deprecated, use --vit_ckpt_path)', required=False)
    parser.add_argument('--tactile_camera_names', nargs='*', default=[], help='Names of tactile sensors for ViT')
    
    # Adapter-related arguments (for ACTJEPAAdapter)
    parser.add_argument('--adapter_hidden_dim', action='store', type=int, default=512,
                        help='Hidden dimension for residual adapter MLP')
    parser.add_argument('--adapter_depth', action='store', type=int, default=3,
                        help='Number of layers in adapter MLP')
    parser.add_argument('--adapter_dropout', action='store', type=float, default=0.1,
                        help='Dropout rate for adapter')
    parser.add_argument('--adapter_scale_init', action='store', type=float, default=0.1,
                        help='Initial value for residual scaling factor')
    parser.add_argument('--adapter_pooling', action='store', type=str, default='attention',
                        choices=['attention', 'mean'],
                        help='Pooling type for aggregating patch tokens')

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args, unknown = parser.parse_known_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    # Build ACTJEPA model if use_vitg is True, otherwise build standard ACT model
    use_vitg = getattr(args, 'use_vitg', False)
    if use_vitg:
        model = build_ACTJEPA_model(args)
        print("Built ACTJEPA model (hybrid RGB+tactile)")
    else:
        model = build_ACT_model(args)
        print("Built ACT model (RGB only)")
    model.cuda()

    # Separate parameters: ViTG encoders should be frozen (no gradients)
    # Backbones (ResNet) get lower learning rate
    # Everything else gets standard learning rate
    use_vitg = getattr(args, 'use_vitg', False)
    has_backbones = any('backbone' in n for n, p in model.named_parameters())
    
    if use_vitg:
        # Exclude ViTG encoders (frozen), separate backbone learning rate if present
        param_dicts = []
        
        # Count ViTG parameters (should all be frozen)
        vitg_params = [(n, p) for n, p in model.named_parameters() if "vitg_encoder" in n]
        vitg_trainable = sum(p.requires_grad for _, p in vitg_params)
        vitg_total = len(vitg_params)
        print(f"ViTG parameters: {vitg_total} total, {vitg_trainable} trainable (should be 0)")
        
        if has_backbones:
            # ResNet backbones with lower LR (hybrid mode)
            backbone_params = [p for n, p in model.named_parameters() 
                              if "backbone" in n and "vitg" not in n and p.requires_grad]
            param_dicts.append({
                "params": backbone_params,
                "lr": args.lr_backbone,
            })
            print(f"Optimizer configured for HYBRID model: {len(backbone_params)} backbone params with LR={args.lr_backbone}")
        else:
            print("Optimizer configured for pure ViTG model (no ResNet backbones)")
        
        # Everything else (transformer, projections) with standard LR
        other_params = [p for n, p in model.named_parameters()
                       if "backbone" not in n and "vitg_encoder" not in n and p.requires_grad]
        param_dicts.append({
            "params": other_params,
            "lr": args.lr,
        })
        print(f"Other trainable params: {len(other_params)} with LR={args.lr}")
    else:
        # Standard configuration with ResNet backbones only
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_ACTJEPAAdapter_model_and_optimizer(args_override):
    """Build ACTJEPAAdapter model with adapter-enhanced ViT encoders"""
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args, unknown = parser.parse_known_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    # Build ACTJEPAAdapter model
    from ModelTrain.detr.models.detr_jepa_adapter import build_jepa_adapter
    model = build_jepa_adapter(args)
    print("Built ACTJEPAAdapter model (hybrid RGB+Adapter-ViT)")
    model.cuda()

    # Separate parameters: 
    # 1. ViT encoder base should be frozen (no gradients)
    # 2. Adapter and pooling should be trainable
    # 3. ResNet backbones get lower learning rate
    # 4. Everything else gets standard learning rate
    
    param_dicts = []
    
    # Count frozen ViT parameters (should not require gradients)
    vitg_base_params = [(n, p) for n, p in model.named_parameters() 
                        if "vitg_encoder" in n and "vitg_base.encoder" in n]
    vitg_base_trainable = sum(p.requires_grad for _, p in vitg_base_params)
    vitg_base_total = len(vitg_base_params)
    print(f"ViT base parameters: {vitg_base_total} total, {vitg_base_trainable} trainable (should be 0)")
    
    # Count adapter parameters (should be trainable)
    adapter_params = [(n, p) for n, p in model.named_parameters() 
                     if "vitg_encoder" in n and ("patch_adapter" in n or "pooling" in n or "query" in n)]
    adapter_trainable = sum(p.requires_grad for _, p in adapter_params)
    adapter_total = len(adapter_params)
    print(f"Adapter parameters: {adapter_total} total, {adapter_trainable} trainable")
    
    # ResNet backbones with lower LR
    has_backbones = any('backbone' in n for n, p in model.named_parameters())
    if has_backbones:
        backbone_params = [p for n, p in model.named_parameters() 
                          if "backbone" in n and "vitg" not in n and p.requires_grad]
        param_dicts.append({
            "params": backbone_params,
            "lr": args.lr_backbone,
        })
        print(f"ResNet backbone params: {len(backbone_params)} with LR={args.lr_backbone}")
    
    # Adapter and pooling with standard LR
    adapter_param_list = [p for n, p in model.named_parameters()
                         if "vitg_encoder" in n and ("patch_adapter" in n or "pooling" in n or "query" in n) 
                         and p.requires_grad]
    if adapter_param_list:
        param_dicts.append({
            "params": adapter_param_list,
            "lr": args.lr,
        })
        print(f"Adapter params: {len(adapter_param_list)} with LR={args.lr}")
    
    # Everything else (transformer, projections, etc.) with standard LR
    other_params = [p for n, p in model.named_parameters()
                   if "backbone" not in n 
                   and "vitg_encoder" not in n 
                   and p.requires_grad]
    param_dicts.append({
        "params": other_params,
        "lr": args.lr,
    })
    print(f"Other trainable params: {len(other_params)} with LR={args.lr}")
    
    # Total trainable parameters
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable:,} ({total_trainable/1e6:.2f}M)")
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args, unknown = parser.parse_known_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

