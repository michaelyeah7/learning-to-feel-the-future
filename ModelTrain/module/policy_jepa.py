"""
ACTJEPA Policy - Hybrid ACT with ResNet for RGB + V-JEPA2 ViTG for tactile
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ModelTrain.detr.main import build_ACT_model_and_optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACTJEPAPolicy(nn.Module):
    """
    ACTJEPA: Hybrid policy with ResNet backbones for RGB cameras and V-JEPA2 ViTG for tactile sensors.
    This policy always uses the hybrid architecture.
    """

    def __init__(self, args_override):
        super().__init__()
        
        # Ensure ACTJEPA requirements are met
        if not args_override.get('use_vitg', False):
            raise ValueError("ACTJEPA requires use_vitg=True")
        if not args_override.get('vitg_ckpt_path'):
            raise ValueError("ACTJEPA requires vitg_ckpt_path to be specified")
        if not args_override.get('tactile_camera_names'):
            raise ValueError("ACTJEPA requires tactile_camera_names to be specified")
        
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.vq = args_override["vq"]
        print(f"ACTJEPA Policy initialized with KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # ACTJEPA always handles list input (RGB and tactile have different resolutions)
        if isinstance(image, list):
            # Normalize each tensor in the list separately
            image = [normalize(img) for img in image]
        else:
            # Fallback: single tensor (shouldn't happen in normal ACTJEPA usage)
            image = normalize(image)
        
        if actions is not None:
            # Training mode
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, env_state, actions, is_pad, vq_sample)
            
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            
            if self.vq:
                loss_dict["vq_discrepancy"] = F.l1_loss(probs, binaries, reduction="mean")
            
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        
        # Inference mode
        a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]
        latent_info, _, _, _, _ = self.model.encode(qpos, actions, is_pad)
        return latent_info

    def serialize(self):
        return self.model.state_dict()

    def deserialize(self, model_dict):
        return self.model.load_state_dict(model_dict)

