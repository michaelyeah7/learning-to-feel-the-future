#!/usr/bin/env python3
"""
Example: Visualizing JEPA-Adapter Embeddings

This example demonstrates how to use the embedding visualization tool
to compare different JEPA checkpoints.
"""

import subprocess
import os

# Configuration
CKPT_DIR = "./ckpt/actjepa_hsa_peg_1107"
CKPT_NAME = "policy_last.ckpt"
EPISODE_IDX = 0
TIMESTEP = 50

# Change to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

print("=" * 60)
print("Visualizing JEPA-Adapter Embeddings")
print("=" * 60)

# Example 1: Visualize with base ViT-L
print("\n1. Generating heatmap with base ViT-L checkpoint...")
subprocess.run([
    "python", "visualize_adapter_embeddings.py",
    "--ckpt_dir", CKPT_DIR,
    "--ckpt_name", CKPT_NAME,
    "--vit_ckpt_path", "./jepa_ckpt/vitl.pt",
    "--episode_idx", str(EPISODE_IDX),
    "--timestep", str(TIMESTEP),
    "--output", "example_embeddings_base.png"
])

# Example 2: Visualize with fine-tuned ViT-L
print("\n2. Generating heatmap with fine-tuned ViT-L checkpoint...")
subprocess.run([
    "python", "visualize_adapter_embeddings.py",
    "--ckpt_dir", CKPT_DIR,
    "--ckpt_name", CKPT_NAME,
    "--vit_ckpt_path", "./jepa_ckpt/vitl_peg_e150.pt",
    "--episode_idx", str(EPISODE_IDX),
    "--timestep", str(TIMESTEP),
    "--output", "example_embeddings_finetuned.png"
])

# Example 3: Visualize different timesteps
print("\n3. Generating heatmaps for different timesteps...")
for t in [10, 50, 100]:
    print(f"   - Timestep {t}...")
    subprocess.run([
        "python", "visualize_adapter_embeddings.py",
        "--ckpt_dir", CKPT_DIR,
        "--ckpt_name", CKPT_NAME,
        "--vit_ckpt_path", "./jepa_ckpt/vitl.pt",
        "--episode_idx", str(EPISODE_IDX),
        "--timestep", str(t),
        "--output", f"example_embeddings_t{t}.png"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\n" + "=" * 60)
print("Visualization complete!")
print("Check the generated PNG files in the project root.")
print("=" * 60)


