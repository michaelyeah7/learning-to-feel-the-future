r"""
Hard Sample Aware (HSA) Contrastive Loss for Tactile-Visual Feature Alignment

This module implements the HSA loss for aligning tactile and visual features:
$$
\mathcal{L}_{\text{HSA-W}} = -\log \frac{\exp(h_\tau \cdot h_w / \kappa)}{\exp(h_\tau \cdot h_w / \kappa) + \sum_{i=1}^{N_k} \exp(h_\tau \cdot h_{w,i}^{\text{neg}} / \kappa)}
$$

where:
- h_tau: tactile features
- h_w: wrist visual features (positive sample)
- h_{w,i}^neg: negative wrist visual features from other samples in the batch
- kappa: temperature parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class HSALoss(nn.Module):
    """Hard Sample Aware contrastive loss for tactile-visual alignment."""
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        """
        Initialize HSA loss.
        
        Args:
            temperature (kappa): Temperature parameter for scaling dot products
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, 
                h_tau: torch.Tensor, 
                h_w: torch.Tensor,
                hard_negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute HSA loss.
        
        Args:
            h_tau: Tactile features, shape (B, D) where B is batch size, D is feature dim
            h_w: Wrist visual features (positive samples), shape (B, D)
            hard_negatives: Optional hard negative samples, shape (B, N_k, D) where N_k is 
                          number of negatives. If None, uses all other samples in batch as negatives.
        
        Returns:
            loss: HSA contrastive loss
        """
        # Normalize features to unit vectors
        h_tau = F.normalize(h_tau, dim=-1)
        h_w = F.normalize(h_w, dim=-1)
        
        batch_size = h_tau.shape[0]
        
        # Compute positive dot products: (B,)
        pos_dot = torch.sum(h_tau * h_w, dim=-1)
        pos_logits = pos_dot / self.temperature
        
        if hard_negatives is not None:
            # Use provided hard negatives
            # hard_negatives shape: (B, N_k, D)
            hard_negatives = F.normalize(hard_negatives, dim=-1)
            
            # Compute negative dot products: (B, N_k)
            neg_dots = torch.einsum('bd,bnd->bn', h_tau, hard_negatives)
            neg_logits = neg_dots / self.temperature
            
        else:
            # Use all other samples in batch as negatives (in-batch negatives)
            # Compute all pairwise dot products: (B, B)
            all_dots = torch.matmul(h_tau, h_w.T)
            all_logits = all_dots / self.temperature
            
            # Create mask to exclude positive pairs (diagonal)
            mask = torch.eye(batch_size, dtype=torch.bool, device=h_tau.device)
            
            # Extract negative logits (off-diagonal elements)
            neg_logits = all_logits.masked_select(~mask).view(batch_size, batch_size - 1)
        
        # Compute loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(exp(pos) + sum(exp(neg)))
        # = -pos + log_sum_exp([pos, neg_1, neg_2, ...])
        
        # Concatenate positive and negative logits
        if hard_negatives is not None:
            logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # (B, 1 + N_k)
        else:
            logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # (B, 1 + (B-1))
        
        # Compute log-sum-exp
        logsumexp = torch.logsumexp(logits, dim=1)  # (B,)
        
        # HSA loss
        loss = -pos_logits + logsumexp  # (B,)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class HSALossWithThirdPerson(nn.Module):
    """
    Extended HSA loss that can also include third-person camera features.
    
    Computes:
    - L_HSA_W: tactile vs wrist features
    - L_HSA_TP: tactile vs third-person features (optional)
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 use_third_person: bool = False,
                 tp_weight: float = 0.5,
                 reduction: str = 'mean'):
        """
        Initialize extended HSA loss.
        
        Args:
            temperature: Temperature parameter
            use_third_person: Whether to also compute loss with third-person features
            tp_weight: Weight for third-person loss term
            reduction: Loss reduction method
        """
        super().__init__()
        self.use_third_person = use_third_person
        self.tp_weight = tp_weight
        
        self.hsa_wrist = HSALoss(temperature, reduction)
        if use_third_person:
            self.hsa_tp = HSALoss(temperature, reduction)
    
    def forward(self,
                h_tau: torch.Tensor,
                h_w: torch.Tensor,
                h_tp: Optional[torch.Tensor] = None,
                hard_negatives_w: Optional[torch.Tensor] = None,
                hard_negatives_tp: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute HSA losses.
        
        Args:
            h_tau: Tactile features (B, D)
            h_w: Wrist visual features (B, D)
            h_tp: Optional third-person visual features (B, D)
            hard_negatives_w: Optional hard negatives for wrist (B, N_k, D)
            hard_negatives_tp: Optional hard negatives for third-person (B, N_k, D)
        
        Returns:
            Dictionary with:
                - 'hsa_wrist': Wrist HSA loss
                - 'hsa_tp': Third-person HSA loss (if enabled)
                - 'hsa_total': Combined HSA loss
        """
        # Compute wrist HSA loss
        loss_w = self.hsa_wrist(h_tau, h_w, hard_negatives_w)
        
        result = {
            'hsa_wrist': loss_w,
            'hsa_total': loss_w
        }
        
        # Optionally compute third-person HSA loss
        if self.use_third_person and h_tp is not None:
            loss_tp = self.hsa_tp(h_tau, h_tp, hard_negatives_tp)
            result['hsa_tp'] = loss_tp
            result['hsa_total'] = loss_w + self.tp_weight * loss_tp
        
        return result


def test_hsa_loss():
    """Test the HSA loss implementation."""
    print("Testing HSA Loss Implementation")
    print("=" * 60)
    
    # Create fake features
    batch_size = 8
    feature_dim = 768
    
    torch.manual_seed(42)
    h_tau = torch.randn(batch_size, feature_dim)
    h_w = torch.randn(batch_size, feature_dim)
    
    # Test basic HSA loss with in-batch negatives
    print("\n[Test 1] Basic HSA loss with in-batch negatives")
    hsa_loss = HSALoss(temperature=0.07)
    loss = hsa_loss(h_tau, h_w)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test with hard negatives
    print("\n[Test 2] HSA loss with explicit hard negatives")
    num_hard_neg = 5
    hard_negatives = torch.randn(batch_size, num_hard_neg, feature_dim)
    loss_with_hard = hsa_loss(h_tau, h_w, hard_negatives)
    print(f"  Loss: {loss_with_hard.item():.4f}")
    
    # Test extended loss with third-person
    print("\n[Test 3] Extended HSA loss with third-person camera")
    h_tp = torch.randn(batch_size, feature_dim)
    extended_loss = HSALossWithThirdPerson(temperature=0.07, use_third_person=True, tp_weight=0.5)
    loss_dict = extended_loss(h_tau, h_w, h_tp)
    print(f"  Wrist loss: {loss_dict['hsa_wrist'].item():.4f}")
    print(f"  Third-person loss: {loss_dict['hsa_tp'].item():.4f}")
    print(f"  Total loss: {loss_dict['hsa_total'].item():.4f}")
    
    # Test gradient flow
    print("\n[Test 4] Gradient flow test")
    h_tau.requires_grad = True
    h_w.requires_grad = True
    loss = hsa_loss(h_tau, h_w)
    loss.backward()
    print(f"  h_tau grad norm: {h_tau.grad.norm().item():.4f}")
    print(f"  h_w grad norm: {h_w.grad.norm().item():.4f}")
    print("  ✓ Gradients computed successfully")
    
    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    test_hsa_loss()

