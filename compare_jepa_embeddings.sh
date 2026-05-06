#!/bin/bash
# Helper script to compare embeddings from different JEPA checkpoints
# This demonstrates how to use visualize_adapter_embeddings.py to compare
# different pretrained ViT weights

CKPT_DIR="./ckpt/actjepa_hsa_peg_1107"
CKPT_NAME="policy_last.ckpt"
EPISODE_IDX=0
TIMESTEP=50

echo "Generating heatmaps for different JEPA checkpoints..."
echo "=================================================="

# Base ViT-L checkpoint
echo "1. Base ViT-L (pretrained on OXE)..."
python visualize_adapter_embeddings.py \
    --ckpt_dir "$CKPT_DIR" \
    --ckpt_name "$CKPT_NAME" \
    --vit_ckpt_path ./jepa_ckpt/vitl.pt \
    --episode_idx "$EPISODE_IDX" \
    --timestep "$TIMESTEP" \
    --output adapter_embeddings_vitl_base.png

# ViT-L fine-tuned on peg task (epoch 0)
echo "2. ViT-L fine-tuned on peg (epoch 0)..."
python visualize_adapter_embeddings.py \
    --ckpt_dir "$CKPT_DIR" \
    --ckpt_name "$CKPT_NAME" \
    --vit_ckpt_path ./jepa_ckpt/vitl_peg_e0.pt \
    --episode_idx "$EPISODE_IDX" \
    --timestep "$TIMESTEP" \
    --output adapter_embeddings_vitl_peg_e0.png

# ViT-L fine-tuned on peg task (epoch 150)
echo "3. ViT-L fine-tuned on peg (epoch 150)..."
python visualize_adapter_embeddings.py \
    --ckpt_dir "$CKPT_DIR" \
    --ckpt_name "$CKPT_NAME" \
    --vit_ckpt_path ./jepa_ckpt/vitl_peg_e150.pt \
    --episode_idx "$EPISODE_IDX" \
    --timestep "$TIMESTEP" \
    --output adapter_embeddings_vitl_peg_e150.png

# ViT-L fine-tuned on peg task (epoch 350)
echo "4. ViT-L fine-tuned on peg (epoch 350)..."
python visualize_adapter_embeddings.py \
    --ckpt_dir "$CKPT_DIR" \
    --ckpt_name "$CKPT_NAME" \
    --vit_ckpt_path ./jepa_ckpt/vitl_peg_e350.pt \
    --episode_idx "$EPISODE_IDX" \
    --timestep "$TIMESTEP" \
    --output adapter_embeddings_vitl_peg_e350.png

echo "=================================================="
echo "Done! Generated heatmaps:"
ls -lh adapter_embeddings_*.png
echo ""
echo "Combined visualizations (with tactile images):"
ls -lh adapter_embeddings_*_combined.png

