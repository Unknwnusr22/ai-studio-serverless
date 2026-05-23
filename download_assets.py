import os
import sys

# Ensure HF cache is permanently routed to the network volume
os.environ["HF_HOME"] = "/runpod-volume/hf_cache"

print("=" * 60)
print("  RunPod Network Volume Pre-Population Script (LTX 2.3 - 10Eros)")
print("=" * 60)

# Create volume directories
models_dir = "/runpod-volume/models"
os.makedirs(models_dir, exist_ok=True)
os.makedirs("/runpod-volume/hf_cache", exist_ok=True)

try:
    import huggingface_hub
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
    import huggingface_hub

from huggingface_hub import hf_hub_download, snapshot_download

# 1. Download 10Eros FP8 Mixed Learned checkpoint
print("\n--- 1. Downloading 10Eros FP8 Mixed Learned checkpoint ---")
dest_file = os.path.join(models_dir, "10Eros_v1-fp8mixed_learned.safetensors")
if os.path.exists(dest_file):
    print(f"File already exists: {dest_file}")
else:
    print("Downloading 10Eros_v1-fp8mixed_learned.safetensors (29.2 GB)...")
    hf_hub_download(
        repo_id="TenStrip/LTX2.3-10Eros",
        filename="10Eros_v1-fp8mixed_learned.safetensors",
        local_dir=models_dir,
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded to: {dest_file}")

# 2. Download DreamFast Gemma-3 Abliterated Text Encoder
print("\n--- 2. Downloading Gemma-3 Abliterated Text Encoder ---")
gemma_dir = os.path.join(models_dir, "gemma-3-12b-it-heretic")
print("Downloading Gemma-3 Heretic Text Encoder and Tokenizer to local folder...")
snapshot_download(
    repo_id="DreamFast/gemma-3-12b-it-heretic",
    local_dir=gemma_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin"] # We only need safetensors and tokenizers/configs
)
print(f"Gemma-3 Text Encoder downloaded successfully to: {gemma_dir}")

# 3. Pre-warm / Cache LTX-2.3 components from HF
print("\n--- 3. Pre-warming LTX-2.3 components ---")
try:
    import diffusers
    import torch
except ImportError:
    print("Installing diffusers & torch...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "diffusers", "transformers", "accelerate", "torch"], check=True)
    import diffusers
    import torch

from diffusers import AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

print("Caching LTX-2.3 Video VAE...")
AutoencoderKLLTX2Video.from_pretrained(
    "diffusers/LTX-2.3-Diffusers",
    subfolder="vae",
    torch_dtype=torch.bfloat16
)

print("Caching LTX-2.3 Connectors...")
LTX2TextConnectors.from_pretrained(
    "diffusers/LTX-2.3-Diffusers",
    subfolder="connectors",
    torch_dtype=torch.bfloat16
)

print("Caching LTX-2.3 Audio VAE...")
AutoencoderKLLTX2Audio.from_pretrained(
    "diffusers/LTX-2.3-Diffusers",
    subfolder="audio_vae",
    torch_dtype=torch.bfloat16
)

print("Caching LTX-2.3 Audio Vocoder...")
LTX2VocoderWithBWE.from_pretrained(
    "diffusers/LTX-2.3-Diffusers",
    subfolder="vocoder",
    torch_dtype=torch.bfloat16
)

print("\n" + "=" * 60)
print("  All LTX 2.3 assets downloaded and cached successfully!")
print("  Cold starts will now load in 0 seconds!")
print("=" * 60)
