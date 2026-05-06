import argparse
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+'/ModelTrain')
sys.path.append(BASE_DIR+'/ModelTrain/detr')
sys.path.append(BASE_DIR+'/robomimic-r2d2')

from module.train_module import train

def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default='./ckpt/dobot_pick_random_1013',required=False)
    parser.add_argument('--task_name', action='store', type=str, default='dobot_pick_random_1013',help='task_name', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=16, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0,required=False)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', default=30000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=2e-5,required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)  # Ignore this parameter and leave the default setting
    parser.add_argument('--eval_every', action='store', type=int, default=100, help='eval_every', required=False)  # Ignore this parameter and leave the default setting
    parser.add_argument('--validate_every', action='store', type=int, default=100, help='validate_every',required=False)
    parser.add_argument('--save_every', action='store', type=int, default=10000, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', default=None, required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')

    parser.add_argument('--kl_weight', action='store', type=int, help='KL divergence weight,recommended set 10 or 100', default=10,required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='The model predicts the length of the output action sequence at a time', default=45,required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200,required=False)
    parser.add_argument('--temporal_agg', action='store_true', default=True)
    parser.add_argument('--no_encoder', action='store_true', default=False)
    
    # Policy class selection
    parser.add_argument('--policy_class', action='store', type=str, default='ACT', 
                        choices=['ACT', 'ACTJEPA', 'ACTJEPAAdapter', 'CNNMLP', 'Diffusion'],
                        help='Policy class to use (ACT for standard, ACTJEPA for hybrid RGB+tactile, ACTJEPAAdapter for learnable adapters)')
    
    # ViT-related arguments (used by ACTJEPA and ACTJEPAAdapter)
    parser.add_argument('--use_vitg', action='store_true', default=False, help='Use ViTG encoder for tactile images (deprecated, use --policy_class ACTJEPA instead)')
    parser.add_argument('--vit_model', action='store', type=str, default='vitg', 
                        choices=['vitg', 'vitl'],
                        help='ViT model type for tactile processing: vitg (1408-dim) or vitl (1024-dim)')
    parser.add_argument('--vit_ckpt_path', action='store', type=str, help='Path to ViT checkpoint file (.pt)', required=False)
    parser.add_argument('--vitg_ckpt_path', action='store', type=str, help='Path to ViTG checkpoint file (deprecated, use --vit_ckpt_path)', required=False)
    
    # Adapter-related arguments (used by ACTJEPAAdapter)
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
    
    # CLIP Encoder arguments
    parser.add_argument('--clip_model', action='store', type=str, default='ViT-B-16',
                        choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14'],
                        help='CLIP model variant for RGB cameras (default: ViT-B-16)')
    parser.add_argument('--clip_pretrained', action='store', type=str, default='openai',
                        help='CLIP pretrained weights (default: openai, can also use laion2b_s34b_b88k, etc.)')
    parser.add_argument('--freeze_clip', action='store_true', default=False,
                        help='Freeze CLIP encoder weights (default: False, i.e., trainable)')
    
    # Text conditioning arguments
    parser.add_argument('--enable_text', action='store_true', default=True,
                        help='Enable text conditioning using CLIP text encoder')
    parser.add_argument('--text_prompt', action='store', type=str, default="Insert the peg into the hole",
                        help='Text prompt for task conditioning (e.g., "insert USB cable", "pick red block")')
    
    # HSA Loss arguments
    parser.add_argument('--enable_hsa', action='store_true', default=False,
                        help='Enable HSA (Hard Sample Aware) loss for tactile-visual alignment')
    parser.add_argument('--hsa_weight', action='store', type=float, default=1.0,
                        help='Weight for HSA loss term (default: 1.0)')
    parser.add_argument('--hsa_temperature', action='store', type=float, default=0.07,
                        help='Temperature parameter for HSA contrastive loss (default: 0.07)')
    parser.add_argument('--hsa_img_size', action='store', type=int, default=224,
                        help='Image size for HSA feature extraction (default: 224)')
    parser.add_argument('--hsa_feature_dim', action='store', type=int, default=768,
                        help='Feature dimension for HSA backbone (default: 768)')
    parser.add_argument('--hsa_num_heads', action='store', type=int, default=12,
                        help='Number of attention heads for HSA backbone (default: 12 for ViT-L, use 16 for ViT-G)')
    parser.add_argument('--robot_type', action='store', type=str, default='Nova 2',
                        choices=['Nova 2', 'Nova 5'],
                        help='Robot type for forward kinematics (default: Nova 2)')
    parser.add_argument('--wrist_camera', action='store', type=str, default='left_wrist',
                        help='Name of the wrist camera for HSA (default: left_wrist)')

    # Use parse_known_args to ignore unknown arguments (e.g., from inference scripts)
    args, unknown = parser.parse_known_args()
    return vars(args)

if __name__ == '__main__':
    args = arg_config()
    train(args)
