"""
RunPod Serverless Handler — LTX 2.3 (10Eros) Image-to-Video
Model loaded from RunPod Network Volume at /workspace/models or /runpod-volume/models
Swaps standard Gemma 3 text encoder with an abliterated (uncensored) Gemma 3 model.
"""

import runpod
import torch
import base64
import io
import os
import tempfile
import traceback
from PIL import Image
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
VOLUME_BASE = os.environ.get("MODEL_DIR", "/workspace/models")

def find_model_path():
    possible_bases = [
        VOLUME_BASE,
        "/workspace/models",
        "/runpod-volume/models",
        "/workspace/models/ltx2.3-10eros",
        "/runpod-volume/models/ltx2.3-10eros",
    ]
    for base in possible_bases:
        path = os.path.join(base, "10eros_v1_bf16.safetensors")
        if os.path.exists(path):
            print(f"[i2v] Found model checkpoint at: {path}")
            return path
    # Fallback to default
    default_path = os.path.join(VOLUME_BASE, "10eros_v1_bf16.safetensors")
    print(f"[i2v] Model checkpoint not found in search paths. Using fallback: {default_path}")
    return default_path

def find_gemma_path():
    possible_dirs = [
        os.path.join(VOLUME_BASE, "gemma-3-12b-it-heretic"),
        os.path.join(VOLUME_BASE, "gemma-3-12b-it-abliterated"),
        "/workspace/models/gemma-3-12b-it-heretic",
        "/runpod-volume/models/gemma-3-12b-it-heretic",
        "/workspace/models/gemma-3-12b-it-abliterated",
        "/runpod-volume/models/gemma-3-12b-it-abliterated",
    ]
    for d in possible_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            if os.path.exists(os.path.join(d, "config.json")):
                print(f"[i2v] Found local Gemma text encoder at: {d}")
                return d
    # Fallback to Hugging Face model hub
    default_hf = "DreamFast/gemma-3-12b-it-heretic"
    print(f"[i2v] Local Gemma not found. Falling back to HF: {default_hf}")
    return default_hf

# ─── Global pipeline reference ───────────────────────────────────────────────
pipe = None


def load_models():
    """Load LTX 2.3 pipeline with abliterated Gemma 3 at container startup."""
    global pipe

    from diffusers import LTX2ImageToVideoPipeline, AutoencoderKLLTX2Video
    from diffusers.models import LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from transformers import Gemma3ForConditionalGeneration, AutoTokenizer

    # Bugfix for diffusers >=0.37.0,<=0.38.0 single-file loading TypeError subtraction bug.
    # In some environments, _get_signature_keys returns optional_kwargs as a list/tuple
    # which causes set subtraction `- optional_kwargs` to crash with TypeError.
    orig_get_signature_keys = LTX2ImageToVideoPipeline._get_signature_keys
    @classmethod
    def patched_get_signature_keys(cls, obj):
        expected_modules, optional_kwargs = orig_get_signature_keys(obj)
        if isinstance(optional_kwargs, (list, tuple)):
            optional_kwargs = set(optional_kwargs)
        return expected_modules, optional_kwargs
    LTX2ImageToVideoPipeline._get_signature_keys = patched_get_signature_keys

    model_path = find_model_path()
    gemma_path = find_gemma_path()

    print(f"[i2v] Loading Gemma-3 Text Encoder from {gemma_path}...")
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path, 
        torch_dtype=torch.bfloat16
    )
    
    print(f"[i2v] Loading Gemma-3 Tokenizer from {gemma_path}...")
    tokenizer = AutoTokenizer.from_pretrained(gemma_path)

    print("[i2v] Transformer will be built and loaded dynamically from the safetensors file...")

    print("[i2v] Loading LTX-2 VAE from Hugging Face...")
    vae = AutoencoderKLLTX2Video.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )
    
    print("[i2v] Loading LTX-2 Connectors from Hugging Face...")
    connectors = LTX2TextConnectors.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="connectors",
        torch_dtype=torch.bfloat16
    )

    print(f"[i2v] Loading LTX 2.3 (10Eros) from {model_path}...")
    pipe = LTX2ImageToVideoPipeline.from_single_file(
        model_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        torch_dtype=torch.bfloat16,
        config="diffusers/LTX-2.3-Diffusers",
        audio_vae=None,
        processor=None,
        vocoder=None,
    )
    
    # Bugfix for Diffusers LTX2ImageToVideoPipeline unconditionally decoding audio
    class DummyAudioConfig:
        sample_rate = 16000
        mel_hop_length = 160
        mel_bins = 64
        latent_channels = 8
        
    class DummyAudioVAE:
        config = DummyAudioConfig()
        mel_compression_ratio = 4
        temporal_compression_ratio = 4
        latents_mean = torch.tensor([0.0])
        latents_std = torch.tensor([1.0])
        dtype = torch.bfloat16
        def decode(self, *args, **kwargs):
            return [torch.zeros((1, 1, 1))]
            
    pipe.audio_vae = DummyAudioVAE()
    pipe.vocoder = lambda x: None

    print("[i2v] Moving LTX 2.3 pipeline to GPU (CUDA) and enforcing bfloat16...")
    pipe.to("cuda", torch.bfloat16)
    print("[i2v] LTX 2.3 loaded and ready.")


def decode_base64_image(b64_string):
    """Decode a base64 string to a PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def frames_to_mp4_base64(frames, fps=24):
    """Convert a list of PIL frames or tensor to base64-encoded MP4."""
    import imageio

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264",
                                     output_params=["-pix_fmt", "yuv420p"])
        for frame in frames:
            if isinstance(frame, Image.Image):
                writer.append_data(np.array(frame))
            elif hasattr(frame, "numpy"):
                arr = frame.cpu().numpy()
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype("uint8")
                writer.append_data(arr)
            else:
                writer.append_data(np.array(frame))
        writer.close()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()

        return base64.b64encode(video_bytes).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def handler(job):
    """
    RunPod handler for LTX 2.3 image-to-video.
    """
    global pipe

    try:
        inp = job["input"]

        # Check for warmup/ping request
        if inp.get("ping", False):
            return {"status": "ready", "message": "LTX 2.3 model is loaded and ready."}

        # ── Required fields ──
        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        image_b64 = inp.get("image_base64")
        if not image_b64:
            return {"error": "image_base64 is required"}

        source_image = decode_base64_image(image_b64)

        # ── Optional params ──
        num_frames = int(inp.get("num_frames", 81))
        guidance = float(inp.get("guidance_scale", 3.0))
        width = int(inp.get("width", 768))
        height = int(inp.get("height", 512))
        num_steps = int(inp.get("num_inference_steps", 30))
        fps = int(inp.get("fps", 24))

        # Enforce LTX constraints: width/height divisible by 32, frames = 8*N+1
        width = (width // 32) * 32
        height = (height // 32) * 32
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1

        # Resize source image
        source_image = source_image.resize((width, height), Image.LANCZOS)

        # ── Run inference ──
        print(f"[i2v] Generating: {width}x{height}, frames={num_frames}, steps={num_steps}")
        with torch.inference_mode():
            result = pipe(
                image=source_image,
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=float(fps),
                guidance_scale=guidance,
                num_inference_steps=num_steps,
            )

        # ── Encode output ──
        frames = result.frames[0]
        video_b64 = frames_to_mp4_base64(frames, fps=fps)

        duration_sec = round(num_frames / fps, 2)
        return {
            "video_base64": video_b64,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": duration_sec,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ─── Load models at startup, then start serverless worker ─────────────────────
print("[i2v] Initializing worker...")
load_models()
print("[i2v] Worker ready.")

runpod.serverless.start({"handler": handler})
