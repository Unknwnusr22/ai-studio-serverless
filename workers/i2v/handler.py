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

    from diffusers import LTX2ImageToVideoPipeline, AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
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

    print("[i2v] Loading LTX-2.3 Audio VAE from Hugging Face...")
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
        "diffusers/LTX-2.3-Diffusers",
        subfolder="audio_vae",
        torch_dtype=torch.bfloat16
    )
    
    print("[i2v] Loading LTX-2.3 Audio Vocoder from Hugging Face...")
    vocoder = LTX2VocoderWithBWE.from_pretrained(
        "diffusers/LTX-2.3-Diffusers",
        subfolder="vocoder",
        torch_dtype=torch.bfloat16
    )

    print(f"[i2v] Loading LTX 2.3 (10Eros) from {model_path} with native audio-visual components...")
    pipe = LTX2ImageToVideoPipeline.from_single_file(
        model_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        connectors=connectors,
        audio_vae=audio_vae,
        vocoder=vocoder,
        torch_dtype=torch.bfloat16,
        config="/app/config",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )

    print("[i2v] Moving LTX 2.3 pipeline to GPU (CUDA) and enforcing bfloat16...")
    pipe.to("cuda", torch.bfloat16)
    print("[i2v] LTX 2.3 loaded and ready with full audio-visual components.")


def decode_base64_image(b64_string):
    """Decode a base64 string to a PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def frames_to_mp4_base64(frames, fps=24, audio=None, sampling_rate=16000):
    """Convert a list of PIL frames or tensor to base64-encoded MP4, optionally with audio."""
    import imageio
    import wave
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        video_path = tmp_video.name
    
    audio_path = None
    output_path = None

    try:
        # 1. Compile video frames
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264",
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

        # 2. If audio is provided, mux them together using ffmpeg
        if audio is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                audio_path = tmp_audio.name
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_output:
                output_path = tmp_output.name
            
            # Convert audio tensor to 16-bit PCM numpy array
            if hasattr(audio, "cpu"):
                audio_arr = audio.float().cpu().numpy()
            else:
                audio_arr = audio
            
            # Flatten to 1D
            audio_arr = audio_arr.flatten()
            
            # Normalize and convert to int16 PCM
            if audio_arr.dtype.kind == 'f':
                audio_arr = np.clip(audio_arr, -1.0, 1.0)
                audio_pcm = (audio_arr * 32767).astype(np.int16)
            else:
                audio_pcm = audio_arr.astype(np.int16)
            
            # Write to WAV file using standard library wave module
            with wave.open(audio_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sampling_rate)
                wav_file.writeframes(audio_pcm.tobytes())

            # Mux video and audio with ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_path
            ]
            print(f"[i2v] Muxing video and audio with cmd: {' '.join(cmd)}")
            mux_result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if mux_result.returncode != 0:
                print(f"[i2v] ffmpeg error: {mux_result.stderr.decode('utf-8')}")
                final_path = video_path
            else:
                print("[i2v] Synchronized audio-visual muxing successful.")
                final_path = output_path
        else:
            final_path = video_path

        with open(final_path, "rb") as f:
            video_bytes = f.read()

        return base64.b64encode(video_bytes).decode("utf-8")
    finally:
        # Clean up temp files
        for p in [video_path, audio_path, output_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


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
        
        # Check and extract generated audio track
        audio = None
        sampling_rate = 16000
        if getattr(result, "audio", None) is not None and len(result.audio) > 0:
            audio = result.audio[0]
            if hasattr(pipe, "vocoder") and hasattr(pipe.vocoder, "config"):
                sampling_rate = getattr(pipe.vocoder.config, "output_sampling_rate", 16000)
            print(f"[i2v] Generated audio track found: shape={getattr(audio, 'shape', None)}, rate={sampling_rate}")
        else:
            print("[i2v] No generated audio track in model output.")

        video_b64 = frames_to_mp4_base64(frames, fps=fps, audio=audio, sampling_rate=sampling_rate)

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
