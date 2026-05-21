"""
RunPod Serverless Handler — Flux.2 Klein Image-to-Image
Supports: IP-Adapter reference images, ControlNet conditioning
Models loaded from RunPod Network Volume at /runpod-volume/models/
"""

import runpod
import torch
import base64
import io
import os
import traceback
from PIL import Image

# ─── Model paths on network volume ───────────────────────────────────────────
VOLUME_BASE = os.environ.get("MODEL_DIR", "/runpod-volume/models")
FLUX_MODEL_PATH = os.path.join(VOLUME_BASE, "flux2-klein")
IP_ADAPTER_PATH = os.path.join(VOLUME_BASE, "ip-adapter-flux2")
CONTROLNET_PATH = os.path.join(VOLUME_BASE, "controlnet-flux2")

# ─── Global pipeline references (loaded once at cold start) ──────────────────
pipe = None
controlnet = None
HAS_IP_ADAPTER = False
HAS_CONTROLNET = False


def load_models():
    """Load all models at container startup."""
    global pipe, controlnet, HAS_IP_ADAPTER, HAS_CONTROLNET

    from diffusers import (
        DiffusionPipeline,
        FluxControlNetModel,
    )

    print(f"[i2i] Loading Flux.2 Klein from {FLUX_MODEL_PATH}...")
    pipe = DiffusionPipeline.from_pretrained(
        FLUX_MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    print("[i2i] Flux.2 Klein loaded.")

    # ── Optional: IP-Adapter ──
    if os.path.isdir(IP_ADAPTER_PATH):
        try:
            print(f"[i2i] Loading IP-Adapter from {IP_ADAPTER_PATH}...")
            pipe.load_ip_adapter(
                IP_ADAPTER_PATH,
                weight_name="ip_adapter.safetensors",
            )
            HAS_IP_ADAPTER = True
            print("[i2i] IP-Adapter loaded.")
        except Exception as e:
            print(f"[i2i] IP-Adapter load failed (non-fatal): {e}")
            HAS_IP_ADAPTER = False
    else:
        print(f"[i2i] No IP-Adapter found at {IP_ADAPTER_PATH}, skipping.")

    # ── Optional: ControlNet ──
    if os.path.isdir(CONTROLNET_PATH):
        try:
            print(f"[i2i] Loading ControlNet from {CONTROLNET_PATH}...")
            controlnet = FluxControlNetModel.from_pretrained(
                CONTROLNET_PATH,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
            HAS_CONTROLNET = True
            print("[i2i] ControlNet loaded.")
        except Exception as e:
            print(f"[i2i] ControlNet load failed (non-fatal): {e}")
            HAS_CONTROLNET = False
    else:
        print(f"[i2i] No ControlNet found at {CONTROLNET_PATH}, skipping.")


def decode_base64_image(b64_string):
    """Decode a base64 string to a PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def encode_image_base64(image, fmt="PNG", quality=95):
    """Encode a PIL Image to a base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess_controlnet(image, mode="canny"):
    """Generate a control image from the source image."""
    import numpy as np

    if mode == "canny":
        try:
            import cv2
            img_array = np.array(image)
            edges = cv2.Canny(img_array, 100, 200)
            control_image = Image.fromarray(edges).convert("RGB")
        except ImportError:
            # Fallback: simple edge detection with PIL
            from PIL import ImageFilter
            control_image = image.filter(ImageFilter.FIND_EDGES)
        return control_image
    elif mode == "depth":
        # Placeholder — in production, use a depth estimation model
        return image.convert("L").convert("RGB")
    elif mode == "pose":
        # Placeholder — in production, use a pose estimation model
        return image
    else:
        return image


def handler(job):
    """
    RunPod handler for Flux.2 Klein image-to-image.

    Input schema:
    {
        "prompt": str,                        # Required
        "image_base64": str,                  # Required — source image
        "negative_prompt": str,               # Optional
        "strength": float,                    # 0.0-1.0, default 0.75
        "num_inference_steps": int,           # default 20
        "guidance_scale": float,              # default 3.5
        "width": int,                         # default: source image width
        "height": int,                        # default: source image height
        "ip_adapter_image_base64": str,       # Optional — reference image for IP-Adapter
        "ip_adapter_scale": float,            # 0.0-1.0, default 0.6
        "controlnet_type": str,               # Optional — "canny", "depth", "pose"
        "controlnet_conditioning_scale": float # 0.0-1.0, default 0.7
    }
    """
    global pipe

    try:
        inp = job["input"]

        # ── Required fields ──
        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        image_b64 = inp.get("image_base64")
        if not image_b64:
            return {"error": "image_base64 is required"}

        source_image = decode_base64_image(image_b64)

        # ── Optional params ──
        negative_prompt = inp.get("negative_prompt", "")
        strength = float(inp.get("strength", 0.75))
        num_steps = int(inp.get("num_inference_steps", 20))
        guidance = float(inp.get("guidance_scale", 3.5))
        width = int(inp.get("width", source_image.width))
        height = int(inp.get("height", source_image.height))

        # Ensure dimensions divisible by 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Resize source to target dims
        source_image = source_image.resize((width, height), Image.LANCZOS)

        # ── Build pipeline kwargs ──
        pipe_kwargs = {
            "prompt": prompt,
            "image": source_image,
            "strength": strength,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
        }

        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        # ── IP-Adapter ──
        ip_b64 = inp.get("ip_adapter_image_base64")
        if ip_b64 and HAS_IP_ADAPTER:
            ip_image = decode_base64_image(ip_b64)
            ip_scale = float(inp.get("ip_adapter_scale", 0.6))
            pipe.set_ip_adapter_scale(ip_scale)
            pipe_kwargs["ip_adapter_image"] = ip_image

        # ── ControlNet ──
        cn_type = inp.get("controlnet_type")
        if cn_type and HAS_CONTROLNET:
            control_image = preprocess_controlnet(source_image, mode=cn_type)
            cn_scale = float(inp.get("controlnet_conditioning_scale", 0.7))
            pipe_kwargs["controlnet_conditioning_scale"] = cn_scale
            pipe_kwargs["control_image"] = control_image

        # ── Run inference ──
        print(f"[i2i] Generating: {width}x{height}, steps={num_steps}, strength={strength}")
        with torch.inference_mode():
            result = pipe(**pipe_kwargs)

        output_image = result.images[0]
        output_b64 = encode_image_base64(output_image)

        return {
            "image_base64": output_b64,
            "width": width,
            "height": height,
            "has_ip_adapter": HAS_IP_ADAPTER and ip_b64 is not None,
            "has_controlnet": HAS_CONTROLNET and cn_type is not None,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ─── Load models at startup, then start serverless worker ─────────────────────
print("[i2i] Initializing worker...")
load_models()
print("[i2i] Worker ready.")

runpod.serverless.start({"handler": handler})
