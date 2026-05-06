import os, torch, numpy as np, pickle, argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import matplotlib
matplotlib.use('Agg')  # Set backend for headless plotting
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from ModelTrain.module.utils import load_data
from ModelTrain.module.utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action
from ModelTrain.module.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from ModelTrain.module.policy_with_hsa import ACTPolicyWithHSA, create_default_hsa_config
from ModelTrain.module.policy_jepa_adapter_with_hsa import ACTJEPAHsa, create_default_hsa_config as create_hsa_config_jepa
import IPython
e = IPython.embed

def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f"qpos_{i}.npy")):
            return i
    else:
        raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def train(args):
    set_seed(1)
    ckpt_dir = args["ckpt_dir"]
    policy_class = args.get("policy_class", "ACT")
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_steps = args["num_steps"]
    eval_every = args["eval_every"]
    validate_every = args["validate_every"]
    save_every = args["save_every"]
    resume_ckpt_path = args["resume_ckpt_path"]
    from ModelTrain.constants import TASK_CONFIGS
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    tactile_camera_names = task_config.get("tactile_camera_names", [])
    stats_dir = task_config.get("stats_dir", None)
    sample_weights = task_config.get("sample_weights", None)
    train_ratio = task_config.get("train_ratio", 0.99)
    name_filter = task_config.get("name_filter", lambda n: True)
    state_dim = 14
    lr_backbone = 1e-05
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr':args["lr"],  'num_queries':args["chunk_size"], 
         'kl_weight':args["kl_weight"], 
         'hidden_dim':args["hidden_dim"], 
         'dim_feedforward':args["dim_feedforward"], 
         'lr_backbone':lr_backbone, 
         'backbone':backbone, 
         'enc_layers':enc_layers, 
         'dec_layers':dec_layers, 
         'nheads':nheads, 
         'camera_names':camera_names, 
         'vq':False, 
         'vq_class':None, 
         'vq_dim':None, 
         'action_dim':16, 
         'no_encoder':args["no_encoder"]}
    elif policy_class == "ACTJEPA":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        # Handle backward compatibility: vit_ckpt_path or vitg_ckpt_path
        vit_ckpt = args.get("vit_ckpt_path") or args.get("vitg_ckpt_path")
        policy_config = {'lr':args["lr"],  'num_queries':args["chunk_size"], 
         'kl_weight':args["kl_weight"], 
         'hidden_dim':args["hidden_dim"], 
         'dim_feedforward':args["dim_feedforward"], 
         'lr_backbone':lr_backbone, 
         'backbone':backbone, 
         'enc_layers':enc_layers, 
         'dec_layers':dec_layers, 
         'nheads':nheads, 
         'camera_names':camera_names, 
         'tactile_camera_names':tactile_camera_names,
         'vq':False, 
         'vq_class':None, 
         'vq_dim':None, 
         'action_dim':16, 
         'no_encoder':args["no_encoder"],
         'use_vitg':True,
         'vitg_ckpt_path':vit_ckpt,
         'vit_model':args.get("vit_model", "vitg")}
    elif policy_class == "ACTJEPAAdapter":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        # Handle backward compatibility: vit_ckpt_path or vitg_ckpt_path
        vit_ckpt = args.get("vit_ckpt_path") or args.get("vitg_ckpt_path")
        policy_config = {'lr':args["lr"],  'num_queries':args["chunk_size"], 
         'kl_weight':args["kl_weight"], 
         'hidden_dim':args["hidden_dim"], 
         'dim_feedforward':args["dim_feedforward"], 
         'lr_backbone':lr_backbone, 
         'backbone':backbone, 
         'enc_layers':enc_layers, 
         'dec_layers':dec_layers, 
         'nheads':nheads, 
         'camera_names':camera_names, 
         'tactile_camera_names':tactile_camera_names,
         'vq':False, 
         'vq_class':None, 
         'vq_dim':None, 
         'action_dim':16, 
         'no_encoder':args["no_encoder"],
         'use_vitg':True,
         'vitg_ckpt_path':vit_ckpt,
         'vit_model':args.get("vit_model", "vitg"),
         'adapter_hidden_dim':args.get("adapter_hidden_dim", 512),
         'adapter_depth':args.get("adapter_depth", 3),
         'adapter_dropout':args.get("adapter_dropout", 0.1),
         'adapter_scale_init':args.get("adapter_scale_init", 0.1),
         'adapter_pooling':args.get("adapter_pooling", "attention")}
    else:
        if policy_class == "Diffusion":
            policy_config = {'lr':args["lr"],  'camera_names':camera_names, 
             'action_dim':16, 
             'observation_horizon':1, 
             'action_horizon':8, 
             'prediction_horizon':args["chunk_size"], 
             'num_queries':args["chunk_size"], 
             'num_inference_timesteps':10, 
             'ema_power':0.75, 
             'vq':False}
        else:
            if policy_class == "CNNMLP":
                policy_config = {'lr':args["lr"], 
                 'lr_backbone':lr_backbone,  'backbone':backbone,  'num_queries':1,  'camera_names':camera_names}
            else:
                raise NotImplementedError
    config = {'num_steps':num_steps,  'eval_every':eval_every, 
     'validate_every':validate_every, 
     'save_every':save_every, 
     'ckpt_dir':ckpt_dir, 
     'resume_ckpt_path':resume_ckpt_path, 
     'episode_len':episode_len, 
     'state_dim':state_dim, 
     'lr':args["lr"], 
     'policy_class':policy_class, 
     'policy_config':policy_config, 
     'task_name':task_name, 
     'seed':args["seed"], 
     'temporal_agg':args["temporal_agg"], 
     'camera_names':camera_names, 
     'load_pretrain':args["load_pretrain"],
     'enable_hsa':args.get("enable_hsa", False),
     'hsa_weight':args.get("hsa_weight", 1.0),
     'hsa_temperature':args.get("hsa_temperature", 0.07),
     'hsa_img_size':args.get("hsa_img_size", 224),
     'hsa_feature_dim':args.get("hsa_feature_dim", 768),
     'hsa_num_heads':args.get("hsa_num_heads", 12),
     'robot_type':args.get("robot_type", "Nova 2"),
     'wrist_camera':args.get("wrist_camera", "left_wrist"),
     'camera_params':args.get("camera_params", None)}
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, "config.pkl")
    expr_name = ckpt_dir.split("/")[-1]
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"Loading data from: {dataset_dir}")
    use_vitg = args.get("use_vitg", False)
    # Use all cameras (RGB + tactile) for data loading
    all_camera_names = camera_names + tactile_camera_names
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, all_camera_names, batch_size_train, batch_size_val, (args["chunk_size"]), (args["skip_mirrored_data"]), (config["load_pretrain"]), policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio, use_vitg=use_vitg, tactile_camera_names=tactile_camera_names)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}")


def make_policy(policy_class, policy_config, hsa_config=None):
    """Create policy with optional HSA loss support."""
    if policy_class == "ACT":
        if hsa_config is not None and hsa_config.get('enable_hsa', False):
            policy = ACTPolicyWithHSA(policy_config, hsa_config)
        else:
            policy = ACTPolicy(policy_config)
    elif policy_class == "ACTJEPA":
        from ModelTrain.module.policy_jepa import ACTJEPAPolicy
        policy = ACTJEPAPolicy(policy_config)
    elif policy_class == "ACTJEPAAdapter":
        from ModelTrain.module.policy_jepa_adapter import ACTJEPAAdapterPolicy
        if hsa_config is not None and hsa_config.get('enable_hsa', False):
            policy = ACTJEPAHsa(policy_config, hsa_config)
        else:
            policy = ACTJEPAAdapterPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "ACTJEPA":
        optimizer = policy.configure_optimizers()
    elif policy_class == "ACTJEPAAdapter":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    elif policy_class == "Diffusion":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, rand_crop_resize=False):
    print("get_image")
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    else:
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        if rand_crop_resize:
            print("rand crop resize is used!")
            original_size = curr_image.shape[-2:]
            ratio = 0.95
            curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2):int(original_size[0] * (1 + ratio) / 2),
             int(original_size[1] * (1 - ratio) / 2):int(original_size[1] * (1 + ratio) / 2)]
            curr_image = curr_image.squeeze(0)
            resize_transform = transforms.Resize(original_size, antialias=True)
            curr_image = resize_transform(curr_image)
            curr_image = curr_image.unsqueeze(0)
        return curr_image


def forward_pass(data, policy, enable_hsa=False):
    """
    Forward pass through policy, handling both standard and HSA modes.
    
    Args:
        data: Batch data (5 items with tactile) or (4 items without tactile)
        policy: Policy model
        enable_hsa: Whether to compute HSA loss (requires tactile data)
    
    Returns:
        Forward dictionary with losses
    """
    # Handle both old format (4 items) and new format (5 items with tactile)
    if len(data) == 5:
        # New format: RGB images, tactile images, qpos, action, is_pad
        rgb_data, tactile_data, qpos_data, action_data, is_pad = data
        
        # Debug: Print shapes to understand data structure
        # print(f"DEBUG: rgb_data type={type(rgb_data)}, shape={rgb_data.shape if isinstance(rgb_data, torch.Tensor) else 'N/A'}")
        # print(f"DEBUG: tactile_data type={type(tactile_data)}, len={len(tactile_data) if isinstance(tactile_data, list) else 'N/A'}")
        # if isinstance(tactile_data, list) and len(tactile_data) > 0:
        #     print(f"DEBUG: tactile_data[0] type={type(tactile_data[0])}")
        #     if isinstance(tactile_data[0], torch.Tensor):
        #         print(f"DEBUG: tactile_data[0] shape={tactile_data[0].shape}")
        #     elif isinstance(tactile_data[0], list):
        #         print(f"DEBUG: tactile_data[0] is list, len={len(tactile_data[0])}")
        #         if len(tactile_data[0]) > 0:
        #             print(f"DEBUG: tactile_data[0][0] type={type(tactile_data[0][0])}, shape={tactile_data[0][0].shape if isinstance(tactile_data[0][0], torch.Tensor) else 'N/A'}")
        
        # Handle tactile_data: could be tensor or list depending on batching
        if isinstance(tactile_data, torch.Tensor):
            # Already a tensor (batch, num_tactile, C, H, W) or (batch, C, H, W)
            if tactile_data.dim() == 4:
                # Single tactile sensor: (batch, C, H, W) -> add camera dim
                tactile_data = tactile_data.unsqueeze(1)  # (batch, 1, C, H, W)
            # Concatenate RGB and tactile along camera dimension
            image_data = torch.cat([rgb_data, tactile_data], dim=1)
        elif tactile_data and len(tactile_data) > 0:
            # It's a list - but DataLoader returns list of already-batched tensors
            # tactile_data is list of (batch, C, H, W) tensors
            
            # Check if first element is already batched (has same batch size as rgb_data)
            if isinstance(tactile_data[0], torch.Tensor) and tactile_data[0].dim() == 4:
                # Each element is (batch, C, H, W), need to add camera dimension
                # Stack along camera dimension: list of (B,C,H,W) -> (B, num_tactile, C, H, W)
                tactile_stacked = torch.stack(tactile_data, dim=1)  # Stack along dim=1 (camera dim)
                # print(f"DEBUG: tactile_stacked shape after stack(dim=1)={tactile_stacked.shape}")
            elif isinstance(tactile_data[0], list):
                # List of lists: (batch) of (num_tactile) of (C, H, W)
                tactile_stacked = torch.stack([torch.stack(batch_tactile) for batch_tactile in tactile_data])
                # print(f"DEBUG: tactile_stacked shape after list-of-lists={tactile_stacked.shape}")
            else:
                # List of per-sample tensors: (batch) of (C, H, W)
                tactile_stacked = torch.stack(tactile_data)  # (batch, C, H, W)
                # print(f"DEBUG: tactile_stacked shape after stack={tactile_stacked.shape}")
                tactile_stacked = tactile_stacked.unsqueeze(1)  # (batch, 1, C, H, W)
                # print(f"DEBUG: tactile_stacked shape after unsqueeze={tactile_stacked.shape}")
            
            # Can't concatenate due to different spatial sizes (480x640 vs 224x224)
            # Pass as list instead - model will handle separately
            image_data = [rgb_data, tactile_stacked]
            # print(f"DEBUG: Passing image_data as list: RGB={rgb_data.shape}, Tactile={tactile_stacked.shape}")
        else:
            # No tactile data
            image_data = rgb_data
    else:
        # Old format: image_data, qpos, action, is_pad
        image_data, qpos_data, action_data, is_pad = data
    
    # Move to CUDA
    if isinstance(image_data, list):
        # List of tensors (hybrid mode with different resolutions)
        image_data = [img.cuda() for img in image_data]
    else:
        image_data = image_data.cuda()
    
    qpos_data, action_data, is_pad = (qpos_data.cuda(), action_data.cuda(), is_pad.cuda())
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config["num_steps"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    eval_every = config["eval_every"]
    validate_every = config["validate_every"]
    save_every = config["save_every"]
    
    # Setup HSA configuration if enabled
    enable_hsa = config.get("enable_hsa", False)
    hsa_config = None
    if enable_hsa:
        # Compute wrist camera index from camera names
        wrist_camera_name = config.get("wrist_camera", "left_wrist")
        camera_names = config['camera_names']
        try:
            wrist_camera_idx = camera_names.index(wrist_camera_name)
        except ValueError:
            print(f"Warning: Wrist camera '{wrist_camera_name}' not found in camera_names {camera_names}, using index 1 (left_wrist)")
            wrist_camera_idx = 1
        
        hsa_config = {
            'enable_hsa': True,
            'hsa_weight': config.get("hsa_weight", 1.0),
            'temperature': config.get("hsa_temperature", 0.07),
            'img_size': config.get("hsa_img_size", 224),
            'feature_dim': config.get("hsa_feature_dim", 768),
            'num_heads': config.get("hsa_num_heads", 12),
            'robot_type': config.get("robot_type", "Nova 2"),
            'wrist_camera': wrist_camera_name,
            'wrist_camera_idx': wrist_camera_idx,
            'tactile_camera_idx': 0,  # Default to first tactile sensor
            'camera_params': config.get("camera_params", None)  # Camera calibration for gripper-aware offset
        }
    
    set_seed(seed)
    policy = make_policy(policy_class, policy_config, hsa_config)
    
    # Print HSA configuration if enabled
    if enable_hsa:
        print("\n" + "="*60)
        print("HSA Loss ENABLED")
        print(f"  Policy Class: {policy_class}")
        print(f"  HSA Weight: {hsa_config['hsa_weight']}")
        print(f"  Temperature: {hsa_config['temperature']}")
        print(f"  Robot Type: {hsa_config['robot_type']}")
        print(f"  Wrist Camera: {hsa_config['wrist_camera']}")
        print("="*60 + "\n")
    
    if config["load_pretrain"]:
        loading_status = policy.deserialize(torch.load(os.path.join("./ckpt/pretrain_all", "policy_step_50000_seed_0.ckpt")))
        print(f"loaded! {loading_status}")
    if config["resume_ckpt_path"] is not None:
        loading_status = policy.deserialize(torch.load(config["resume_ckpt_path"]))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    optimizer = make_optimizer(policy_class, policy)
    policy.cuda()
    min_val_loss = np.inf
    best_ckpt_info = None
    train_dataloader = repeater(train_dataloader)
    train_loss = []
    val_loss = []
    train_hsa = []  # Track HSA loss
    val_hsa = []    # Track validation HSA loss
    last_time = time.time()
    start_time = last_time
    for step in tqdm(range(num_steps + 1)):
        if step % validate_every == 0:
            print("validating")
            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy, enable_hsa=enable_hsa)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break
                    validation_summary = compute_dict_mean(validation_dicts)
                    epoch_val_loss = validation_summary["loss"].mean()
                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))

            for k in list(validation_summary.keys()):
                validation_summary[f"val_{k}"] = validation_summary.pop(k)
            else:
                print(f"Val loss:   {epoch_val_loss:.5f}")
                val_loss.append(float(epoch_val_loss.item()))
                
                # Track HSA validation loss if enabled
                if enable_hsa and 'val_hsa_total' in validation_summary:
                    val_hsa.append(float(validation_summary['val_hsa_total'].mean().item()))
                
                # Print HSA losses prominently if enabled
                if enable_hsa:
                    hsa_keys = [k for k in validation_summary.keys() if 'hsa' in k.lower()]
                    if hsa_keys:
                        print("  HSA Losses:", end=" ")
                        for k in hsa_keys:
                            print(f"{k}: {validation_summary[k].mean().item():.3f}", end=" ")
                        print()
                
                # Print all validation metrics
                summary_string = ""
                for k, v in validation_summary.items():
                    summary_string += f"{k}: {v.mean().item():.3f} "
                else:
                    print(summary_string)

        if step > 0:
            if step % eval_every == 0:
                ckpt_name = f"policy_step_{step}_seed_{seed}.ckpt"
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy, enable_hsa=enable_hsa)
        loss = forward_dict["loss"]
        loss.mean().backward()
        optimizer.step()
        train_loss.append(float(loss.mean().item()))
        
        # Track HSA training loss if enabled
        if enable_hsa and 'hsa_total' in forward_dict:
            train_hsa.append(float(forward_dict['hsa_total'].mean().item()))
        
        # Print training loss periodically (every 100 steps)
        if step % 100 == 0 and step > 0:
            train_summary = f"Step {step} - Train loss: {loss.mean().item():.5f}"
            if enable_hsa and 'hsa_wrist' in forward_dict:
                train_summary += f" | L1: {forward_dict['l1'].mean().item():.3f}"
                train_summary += f" | KL: {forward_dict['kl'].mean().item():.3f}"
                train_summary += f" | HSA_wrist: {forward_dict['hsa_wrist'].mean().item():.3f}"
                if 'hsa_total' in forward_dict:
                    train_summary += f" | HSA_total: {forward_dict['hsa_total'].mean().item():.3f}"
            print(train_summary)
        
        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_step_{step}_seed_{seed}.ckpt")
            torch.save(policy.serialize(), ckpt_path)
        cur_time = time.time()
        last_time = cur_time
    else:
        print("train all time:", cur_time - start_time)
        ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
        torch.save(policy.serialize(), ckpt_path)
        best_step, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f"policy_step_{best_step}_seed_{seed}.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}")
        
        # Plot training loss with smoothing
        plt.figure(figsize=(12, 6))
        plt.plot(train_loss, label="Training Loss (raw)", color='blue', alpha=0.2, linewidth=0.5)
        
        # Add smoothed line
        if len(train_loss) > 100:
            window = min(100, len(train_loss) // 10)
            smoothed = np.convolve(train_loss, np.ones(window)/window, mode='valid')
            smooth_steps = list(range(window//2, len(train_loss) - window//2))
            plt.plot(smooth_steps, smoothed, label=f"Training Loss (smoothed)", 
                    color='darkblue', linewidth=2)
        
        plt.title("Training Loss Over Steps")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(ckpt_dir + "/train_loss.png", dpi=150)
        plt.close()
        
        # Plot validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(val_loss, label="Validation Loss", color='green', linewidth=2, marker='o', markersize=4)
        plt.title("Validation Loss Over Steps")
        plt.xlabel("Validation Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(ckpt_dir + "/val_loss.png", dpi=150)
        plt.close()
        
        # Plot HSA losses if available
        if enable_hsa and len(train_hsa) > 0:
            # Training HSA loss with smoothing
            plt.figure(figsize=(12, 6))
            
            # Raw data with transparency
            plt.plot(train_hsa, label="HSA Loss (raw)", color='orange', alpha=0.2, linewidth=0.5)
            
            # Smoothed data using moving average
            window_size = min(100, len(train_hsa) // 10)  # Adaptive window
            if len(train_hsa) > window_size:
                smoothed = np.convolve(train_hsa, np.ones(window_size)/window_size, mode='valid')
                smoothed_steps = list(range(window_size//2, len(train_hsa) - window_size//2))
                plt.plot(smoothed_steps, smoothed, label=f"HSA Loss (smoothed, window={window_size})", 
                        color='darkorange', linewidth=2)
            
            plt.title("HSA Training Loss Over Steps")
            plt.xlabel("Step")
            plt.ylabel("HSA Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(ckpt_dir + "/train_hsa_loss.png", dpi=150)
            plt.close()
            
            # Validation HSA loss
            if len(val_hsa) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(val_hsa, label="Validation HSA Loss", color='red')
                plt.title("HSA Validation Loss Over Steps")
                plt.xlabel("Validation Step")
                plt.ylabel("HSA Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig(ckpt_dir + "/val_hsa_loss.png")
                plt.close()
            
            # Combined plot: Training and Validation HSA
            if len(val_hsa) > 0:
                plt.figure(figsize=(12, 6))
                
                # Raw training data (very transparent)
                plt.plot(train_hsa, color='orange', alpha=0.15, linewidth=0.5, label='Training (raw)')
                
                # Smoothed training data
                window_size = min(100, len(train_hsa) // 10)
                if len(train_hsa) > window_size:
                    smoothed = np.convolve(train_hsa, np.ones(window_size)/window_size, mode='valid')
                    smoothed_steps = list(range(window_size//2, len(train_hsa) - window_size//2))
                    plt.plot(smoothed_steps, smoothed, color='darkorange', linewidth=2, 
                            label=f'Training (smoothed)')
                
                # Validation data
                val_steps = [i * validate_every for i in range(len(val_hsa))]
                plt.plot(val_steps, val_hsa, color='red', linewidth=2.5, 
                        marker='o', markersize=4, label='Validation')
                
                plt.title("HSA Loss: Training vs Validation")
                plt.xlabel("Step")
                plt.ylabel("HSA Loss")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(ckpt_dir + "/hsa_loss_combined.png", dpi=150)
                plt.close()
            
            print(f"HSA loss plots saved to {ckpt_dir}/")
            if len(train_hsa) > 100:
                print(f"  Initial HSA: {train_hsa[0]:.3f} → Final HSA: {train_hsa[-1]:.3f}")
        
        return best_ckpt_info


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        else:
            print(f"Epoch {epoch} done")
            epoch += 1

# okay decompiling train_module.pyc
