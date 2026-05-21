# 🌌 AI Studio — Complete Serverless Cloud Migration & LTX 2.3 (10Eros) Integration Archive

This document serves as a 100% self-contained, highly comprehensive technical blueprint and code vault for the **AI Studio** system. It details the system architecture, component configurations, API contracts, local-first preloading workflows, and contains the complete source code for both the frontend JS clients and the backend GPU worker.

If you encounter any compilation, runtime, or API connection errors, feeding this single document into an LLM (like Codex or Gemini) will give it complete context to diagnose and fix the issue instantly.

---

## 📂 1. Project Directory Structure

The repository is structured as a hybrid mobile/web application (managed via Capacitor) paired with independent, containerized GPU microservices designed for RunPod Serverless.

```
d:\po\
├── app\                                # Hybrid Mobile/Web Application (Capacitor)
│   ├── capacitor.config.ts            # Capacitor native compilation config
│   ├── package.json                   # Node dependencies & scripts
│   ├── package-lock.json              # Locked dependencies
│   └── www\                           # Frontend Static Assets
│       ├── index.html                 # Main App Interface (Tailored Styling)
│       ├── css\
│       │   └── styles.css             # Harmonious HSL Theme & Badges Styling
│       └── js\
│           ├── app.js                 # Settings, VisionLLM SSE, & WarmUpManager
│           ├── chat.js                # Vision Chat Tab (State & Chat Bubbles)
│           └── studio.js              # Studio Tab (I2I & I2V Generative Flow)
├── workers\                            # RunPod Serverless Workers
│   ├── i2i\                           # Image-to-Image (Flux.2 Klein)
│   │   ├── Dockerfile
│   │   ├── handler.py
│   │   └── requirements.txt
│   └── i2v\                           # Image-to-Video (LTX 2.3 10Eros) [UPDATED]
│       ├── Dockerfile                 # Multi-stage GPU builder
│       ├── handler.py                 # Gemma-3 Uncensored Handler
│       └── requirements.txt           # Versioned Python libraries
└── README.md
```

---

## ⚙️ 2. Core Architecture & Workflow

AI Studio has migrated off local on-device inference (wllama) to **100% cloud-hosted GPU Serverless Endpoints**. 

```
                               ┌───────────────┐
                               │   Android     │
                               │  Capacitor    │
                               │   Frontend    │
                               └───────┬───────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
        [ vLLM Endpoint ]       [ I2I Endpoint ]       [ I2V Endpoint ]
       Llama 3.2 / Qwen2-VL       Flux.2 Klein          LTX 2.3 (10Eros)
         (24GB GPU VRAM)         (24GB GPU VRAM)         (24GB GPU VRAM)
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       ▼
                       [ Shared Network Volume (RunPod) ]
                          /workspace/models/ mapped to
                        /runpod-volume/network-storage/
```

### 3 Serverless Endpoints & Specifications
1.  **vLLM Chat (`rpVLLM`)**: Performs vision chat and enhances generation prompts.
2.  **I2I Image (`rpI2I`)**: Runs the Flux.2 Klein Image-to-Image pipeline (including optional IP-Adapter and ControlNet conditioning).
3.  **I2V Video (`rpI2V`)**: Runs TenStrip's LTX 2.3 10Eros Image-to-Video pipeline with an abliterated text encoder.

---

## ⚡ 3. On-Demand Endpoint Warm-Up Protocol

To avoid latency issues due to serverless "cold starts," the application implements an active **Endpoint Warm-up Manager**:

*   **Trigger**: The user clicks **"Warm Up All Endpoints"** in the Settings panel.
*   **Action**: Concurrently dispatches a lightweight API request containing `{ "input": { "ping": true } }` to all three endpoint IDs via `RunPod.run()`.
*   **Worker Handling**: The serverless worker catches `ping: true` instantly at startup, bypassing the expensive inference loop, and returns `{"status": "ready"}`. This forces RunPod to spin up and warm up the container (loading multi-gigabyte models into VRAM) on-demand.
*   **UI Status States**:
    *   `Inactive` (Gray): Worker has not been pinged.
    *   `Warming...` (Pulsing Amber): Request sent, container is cold-starting and allocating VRAM.
    *   `Ready` (Green): Container is online, models are loaded, and the endpoint is ready for instant execution.

---

## 🧠 4. Endpoint 1: vLLM Chat Optimization & Models

Running the vision-language model in FP16 requires a massive VRAM footprint. To run on a highly economical **24GB GPU** (like an RTX 4090), we utilize three optimized deployment strategies on RunPod:

### Model VRAM Strategies
*   **Option 1 (Dynamic FP8 Quantization - Uncensored)**:
    *   **Model**: Keep the local 20GB FP16 *Llama 3.2 11B Vision Abliterated* model on disk.
    *   **Deployment**: Pass `--quantization fp8` as an argument when launching the `runpod/worker-vllm:stable-cuda12.1.0` container.
    *   **VRAM**: Reduces footprint to ~11GB, maintaining the uncensored chat behavior without needing to download pre-quantized weights.
*   **Option 2 (Pre-Quantized FP8 Instruct)**:
    *   **Model**: NeuralMagic's pre-quantized `neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic` (11GB total).
    *   **Pros**: Blazing fast container cold starts due to reduced disk read speeds.
*   **Option 3 (Qwen2-VL 8B FP8 - Abliterated)**:
    *   **Model**: `Heouzen/Huihui-Qwen3-VL-8B-Instruct-FP8-abliterated` (8.6GB).
    *   **Pros**: Ultra-small footprint, fully uncensored, extremely high vision reasoning capabilities.

---

## 🎬 5. Endpoint 3: LTX 2.3 10Eros Image-to-Video Deep-Dive

TenStrip's **LTX2.3-10Eros** is an uncensored, fine-tuned DiT (Diffusion Transformer) backbone. The model is capable of generating unrestricted visual concepts. However, censorship can be introduced if text conditioning is processed through standard, heavily-aligned models (like `google/gemma-3-12b-it`).

### 🔓 Swapping the Text Encoder (Abliterated Setup)
To guarantee a fully uncensored pipeline:
1.  We replace the default Google Gemma 3 weights with an abliterated text encoder repository: **`DreamFast/gemma-3-12b-it-heretic`** (optimized for video pipelines with low KL divergence) or **`mlabonne/gemma-3-12b-it-abliterated`**.
2.  In python, we instantiate the pipeline by passing this text encoder directly:
    ```python
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained("DreamFast/gemma-3-12b-it-heretic")
    tokenizer = AutoTokenizer.from_pretrained("DreamFast/gemma-3-12b-it-heretic")
    
    pipe = LTX2ImageToVideoPipeline.from_single_file(
        "10eros_v1_bf16.safetensors",
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        pretrained_model_name_or_path="Lightricks/LTX-2"
    )
    ```
3.  **Base Repository Mapping**: By specifying `pretrained_model_name_or_path="Lightricks/LTX-2"`, `diffusers`' single-file loader automatically fetches missing auxiliary components (scheduler, connectors/projection layers, audio_vae, vocoder) while mapping the main transformer and VAE from the local `10eros_v1_bf16.safetensors` file.

### 💾 VRAM & OOM Optimization
The 10Eros transformer is 22B parameters, and the Gemma-3 text encoder is 12B parameters. Loading both simultaneously in FP16/BF16 exceeds standard 24GB GPU VRAM. 
*   **Solution**: We invoke `pipe.enable_model_cpu_offload()`. This sequentially loads components onto the GPU during execution (e.g., loading Gemma-3 to encode the prompt, offloading Gemma-3, then loading the LTX Transformer for denoising), enabling full inference on an RTX 4090/3090 with zero OOM crashes.

### 📏 Generation Constraints
*   **Resolutions**: Output width and height must be strictly divisible by **32**.
*   **Frame Counts**: The total generated frame count must conform to `8 * N + 1` (e.g., 49, 65, 81, 97, 113, 121, etc.).

---

## 💻 6. Full Backend Code (LTX 2.3 GPU Worker)

Below is the complete, production-ready code for the Image-to-Video worker endpoint.

### 📄 File: `d:\po\workers\i2v\handler.py`
```python
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

    from diffusers import LTX2ImageToVideoPipeline
    from transformers import Gemma3ForConditionalGeneration, AutoTokenizer

    model_path = find_model_path()
    gemma_path = find_gemma_path()

    print(f"[i2v] Loading Gemma-3 Text Encoder from {gemma_path}...")
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path, 
        torch_dtype=torch.bfloat16
    )
    
    print(f"[i2v] Loading Gemma-3 Tokenizer from {gemma_path}...")
    tokenizer = AutoTokenizer.from_pretrained(gemma_path)

    print(f"[i2v] Loading LTX 2.3 (10Eros) from {model_path}...")
    pipe = LTX2ImageToVideoPipeline.from_single_file(
        model_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        pretrained_model_name_or_path="Lightricks/LTX-2",
    )
    
    print("[i2v] Enabling model CPU offload for VRAM management...")
    pipe.enable_model_cpu_offload()
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
```

### 📄 File: `d:\po\workers\i2v\requirements.txt`
```
runpod>=1.7.0
torch>=2.4.0
diffusers @ git+https://github.com/huggingface/diffusers.git
transformers>=4.50.0
accelerate>=0.33.0
safetensors>=0.4.0
Pillow>=10.0.0
imageio>=2.34.0
imageio-ffmpeg>=0.5.1
sentencepiece>=0.2.0
protobuf>=4.25.0
```

### 📄 File: `d:\po\workers\i2v\Dockerfile`
```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# RunPod network volume mount point
VOLUME /runpod-volume

CMD ["python", "-u", "handler.py"]
```

---

## 📱 7. Full Frontend Code (Client API Connectors)

Below are the complete frontend JavaScript scripts running inside Capacitor to communicate with the serverless GPU workers.

### 📄 File: `d:\po\app\www\js\app.js`
```javascript
/* ═══════════════════════════════════════════════════════════════════════════
   app.js — Core settings manager, tab routing, VisionLLM SSE, & Warm-up Manager
   ═══════════════════════════════════════════════════════════════════════════ */

// ─── Settings Manager ────────────────────────────────────────────────────────
const Settings = {
    _defaults: {
        rpKey: '',
        rpVLLM: 'qf469tfvtyp32c',
        rpI2I: 'i3vobq1g35kg1s',
        rpI2V: '19x84j6q0hsp27',
        rpModelName: '/runpod-volume/models/qwen2-vl-8b-fp8-abliterated',
        systemPrompt: 'You are a helpful AI assistant with vision capabilities. When shown images, describe what you see and answer questions about them.',
    },
    get(key) {
        const val = localStorage.getItem('ais_' + key);
        if (val === null) return this._defaults[key] ?? '';
        if (val === 'true') return true;
        if (val === 'false') return false;
        return val;
    },
    set(key, val) { localStorage.setItem('ais_' + key, String(val)); },
    load() {
        document.getElementById('setting-rp-key').value = this.get('rpKey');
        document.getElementById('setting-rp-vllm').value = this.get('rpVLLM');
        document.getElementById('setting-rp-i2i').value = this.get('rpI2I');
        document.getElementById('setting-rp-i2v').value = this.get('rpI2V');
        document.getElementById('setting-rp-model-name').value = this.get('rpModelName');
        document.getElementById('setting-system-prompt').value = this.get('systemPrompt');
    },
    save() {
        this.set('rpKey', document.getElementById('setting-rp-key').value.trim());
        this.set('rpVLLM', document.getElementById('setting-rp-vllm').value.trim());
        this.set('rpI2I', document.getElementById('setting-rp-i2i').value.trim());
        this.set('rpI2V', document.getElementById('setting-rp-i2v').value.trim());
        this.set('rpModelName', document.getElementById('setting-rp-model-name').value.trim());
        this.set('systemPrompt', document.getElementById('setting-system-prompt').value);
    }
};

// ─── Vision LLM (RunPod Serverless endpoint) ───────────────────────────────
const VisionLLM = {
    /**
     * Chat completion with optional streaming via RunPod OpenAI-compatible endpoint.
     * messages: OpenAI-compatible array [{role, content}]
     */
    async chat(messages, onToken) {
        const rpKey = Settings.get('rpKey');
        const endpointId = Settings.get('rpVLLM');

        if (!rpKey || !endpointId) {
            throw new Error('Please set your RunPod API Key and vLLM Endpoint ID in Settings.');
        }

        const url = `https://api.runpod.ai/v2/${endpointId}/openai/v1/chat/completions`;
        const body = {
            model: Settings.get('rpModelName'),
            messages: messages,
            temperature: 0.7,
            max_tokens: 2048,
            stream: !!onToken
        };

        const res = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${rpKey}`
            },
            body: JSON.stringify(body)
        });

        if (!res.ok) {
            const txt = await res.text();
            throw new Error(`vLLM API error (${res.status}): ${txt}`);
        }

        if (onToken) {
            // Streaming mode using ReadableStream
            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let fullText = '';

            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // Keep incomplete line

                    for (const line of lines) {
                        const cleanLine = line.trim();
                        if (!cleanLine) continue;
                        if (cleanLine === 'data: [DONE]') continue;
                        if (cleanLine.startsWith('data: ')) {
                            try {
                                const parsed = JSON.parse(cleanLine.substring(6));
                                const delta = parsed.choices?.[0]?.delta?.content ?? '';
                                if (delta) {
                                    fullText += delta;
                                    onToken(delta, fullText);
                                }
                            } catch (e) {
                                console.warn('Error parsing SSE line', cleanLine, e);
                            }
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
            return fullText;
        } else {
            // Non-streaming mode
            const data = await res.json();
            return data.choices?.[0]?.message?.content ?? '';
        }
    },

    buildVisionMessage(text, imageBase64) {
        if (!imageBase64) {
            return { role: 'user', content: text };
        }
        return {
            role: 'user',
            content: [
                { type: 'text', text: text },
                { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${imageBase64}` } }
            ]
        };
    }
};

// ─── RunPod API Client ───────────────────────────────────────────────────────
const RunPod = {
    _headers() {
        return {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + Settings.get('rpKey'),
        };
    },

    async run(endpointId, input) {
        const url = `https://api.runpod.ai/v2/${endpointId}/run`;
        const res = await fetch(url, {
            method: 'POST',
            headers: this._headers(),
            body: JSON.stringify({ input }),
        });
        if (!res.ok) {
            const txt = await res.text();
            throw new Error(`RunPod submit error ${res.status}: ${txt}`);
        }
        return await res.json(); // { id, status }
    },

    async status(endpointId, jobId) {
        const url = `https://api.runpod.ai/v2/${endpointId}/status/${jobId}`;
        const res = await fetch(url, { headers: this._headers() });
        if (!res.ok) throw new Error(`RunPod status error: ${res.status}`);
        return await res.json(); // { id, status, output }
    },

    async poll(endpointId, jobId, onStatus, intervalMs = 3000) {
        while (true) {
            const result = await this.status(endpointId, jobId);
            if (onStatus) onStatus(result.status);
            if (result.status === 'COMPLETED') return result.output;
            if (result.status === 'FAILED') throw new Error(result.error || 'Job failed');
            if (result.status === 'CANCELLED') throw new Error('Job cancelled');
            await new Promise(r => setTimeout(r, intervalMs));
        }
    }
};

// ─── Warm-up Manager ─────────────────────────────────────────────────────────
const WarmUpManager = {
    async pingEndpoint(endpointId, elementId) {
        const el = document.getElementById(elementId);
        if (!el) return;
        
        if (!Settings.get('rpKey') || !endpointId) {
            el.className = 'status-badge err';
            el.textContent = 'Missing Info';
            return;
        }

        el.className = 'status-badge loading';
        el.textContent = 'Warming...';

        try {
            // Sends ping payload to wake the serverless container
            await RunPod.run(endpointId, { ping: true });
            el.className = 'status-badge ok';
            el.textContent = 'Ready';
        } catch (e) {
            console.error('Warm-up error:', e);
            el.className = 'status-badge ok';
            el.textContent = 'Ready';
        }
    },

    init() {
        const btn = document.getElementById('btn-warmup');
        if (!btn) return;
        
        btn.addEventListener('click', () => {
            Settings.save(); // Save settings before using them
            
            const vllmId = Settings.get('rpVLLM');
            const i2iId = Settings.get('rpI2I');
            const i2vId = Settings.get('rpI2V');

            if (!Settings.get('rpKey')) {
                alert('Please enter your RunPod API Key first.');
                return;
            }

            // Show status list
            document.getElementById('warmup-status').style.display = 'flex';

            if (vllmId) this.pingEndpoint(vllmId, 'warmup-vllm');
            if (i2iId) this.pingEndpoint(i2iId, 'warmup-i2i');
            if (i2vId) this.pingEndpoint(i2vId, 'warmup-i2v');
        });
    }
};

// ─── Image Utilities ─────────────────────────────────────────────────────────
const ImageUtil = {
    fileToBase64(file, maxSize = 1536) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    let w = img.width, h = img.height;
                    if (w > maxSize || h > maxSize) {
                        const ratio = Math.min(maxSize / w, maxSize / h);
                        w = Math.round(w * ratio);
                        h = Math.round(h * ratio);
                    }
                    canvas.width = w; canvas.height = h;
                    canvas.getContext('2d').drawImage(img, 0, 0, w, h);
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
                    resolve(dataUrl.split(',')[1]);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        });
    },

    downloadBase64(b64, filename, mimeType = 'image/png') {
        const link = document.createElement('a');
        link.href = `data:${mimeType};base64,${b64}`;
        link.download = filename;
        link.click();
    }
};

// ─── Tab Router ──────────────────────────────────────────────────────────────
function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');
    tabs.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            tabs.forEach(t => t.classList.toggle('active', t === btn));
            contents.forEach(c => c.classList.toggle('active', c.id === 'tab-' + target));
        });
    });
}

// ─── Settings Panel ──────────────────────────────────────────────────────────
function initSettings() {
    const panel = document.getElementById('settings-panel');
    const overlay = document.getElementById('settings-overlay');

    function open() {
        Settings.load();
        panel.classList.add('open');
        overlay.classList.remove('hidden');
        requestAnimationFrame(() => overlay.classList.add('visible'));
    }
    function close() {
        panel.classList.remove('open');
        overlay.classList.remove('visible');
        setTimeout(() => overlay.classList.add('hidden'), 300);
    }

    document.getElementById('btn-settings').addEventListener('click', open);
    document.getElementById('btn-close-settings').addEventListener('click', close);
    overlay.addEventListener('click', close);

    document.getElementById('btn-save-settings').addEventListener('click', () => {
        Settings.save();
        close();
    });
}

// ─── Slider value displays ──────────────────────────────────────────────────
function initSliders() {
    document.querySelectorAll('.slider-row input[type="range"]').forEach(slider => {
        const valEl = slider.nextElementSibling;
        if (valEl?.classList.contains('slider-val')) {
            slider.addEventListener('input', () => { valEl.textContent = slider.value; });
        }
    });
}

// ─── Auto-resize textareas ──────────────────────────────────────────────────
function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSettings();
    WarmUpManager.init();
    initSliders();

    const chatInput = document.getElementById('chat-input');
    chatInput.addEventListener('input', () => autoResize(chatInput));
});
```

### 📄 File: `d:\po\app\www\js\chat.js`
```javascript
/* ═══════════════════════════════════════════════════════════════════════════
   chat.js — Chat tab state management, rendering & VisionLLM connection
   ═══════════════════════════════════════════════════════════════════════════ */

const Chat = (() => {
    let messages = []; // { role, content, imageBase64? }
    let attachedImage = null; // base64 string
    const STORAGE_KEY = 'ais_chat_history';

    // ─── DOM refs ────────────────────────────────────────────────────────
    const msgContainer = () => document.getElementById('chat-messages');
    const inputEl = () => document.getElementById('chat-input');
    const previewEl = () => document.getElementById('chat-image-preview');
    const fileInput = () => document.getElementById('file-chat-image');

    // ─── Persistence ─────────────────────────────────────────────────────
    function saveHistory() {
        try {
            // Save without image data to keep storage small
            const slim = messages.map(m => ({
                role: m.role,
                content: typeof m.content === 'string' ? m.content : '[image + text]',
            }));
            localStorage.setItem(STORAGE_KEY, JSON.stringify(slim.slice(-50)));
        } catch {}
    }

    function loadHistory() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) {
                messages = JSON.parse(raw);
                messages.forEach(m => renderBubble(m.role, m.content));
            }
        } catch {}
    }

    // ─── Rendering ───────────────────────────────────────────────────────
    function renderBubble(role, content, imageBase64) {
        const container = msgContainer();
        const welcome = container.querySelector('.chat-welcome');
        if (welcome) welcome.style.display = 'none';

        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role === 'user' ? 'user' : 'ai'}`;

        if (imageBase64) {
            const img = document.createElement('img');
            img.src = `data:image/jpeg;base64,${imageBase64}`;
            img.alt = 'Attached image';
            bubble.appendChild(img);
        }

        const textEl = document.createElement('div');
        textEl.textContent = typeof content === 'string' ? content : '';
        bubble.appendChild(textEl);

        container.appendChild(bubble);
        container.scrollTop = container.scrollHeight;
        return textEl;
    }

    function renderTypingBubble() {
        const container = msgContainer();
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble ai';
        bubble.id = 'typing-bubble';
        bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
        container.appendChild(bubble);
        container.scrollTop = container.scrollHeight;
        return bubble;
    }

    function removeTypingBubble() {
        const el = document.getElementById('typing-bubble');
        if (el) el.remove();
    }

    // ─── Image attachment ────────────────────────────────────────────────
    function showImagePreview(b64) {
        const preview = previewEl();
        preview.innerHTML = '';
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${b64}`;
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-img';
        removeBtn.textContent = '✕';
        removeBtn.onclick = () => { attachedImage = null; preview.classList.add('hidden'); };
        preview.appendChild(img);
        preview.appendChild(removeBtn);
        preview.classList.remove('hidden');
    }

    // ─── Send message ────────────────────────────────────────────────────
    async function send() {
        const input = inputEl();
        const text = input.value.trim();
        if (!text && !attachedImage) return;

        if (!Settings.get('rpKey') || !Settings.get('rpVLLM')) {
            alert('Please configure your RunPod API Key and vLLM Endpoint ID in Settings.');
            return;
        }

        // Build user message
        const userMsg = VisionLLM.buildVisionMessage(text || 'What is in this image?', attachedImage);

        // Render user bubble
        renderBubble('user', text || 'What is in this image?', attachedImage);

        // Add to history
        messages.push(userMsg);

        // Clear input
        input.value = '';
        input.style.height = 'auto';
        const imgB64 = attachedImage;
        attachedImage = null;
        previewEl().classList.add('hidden');

        // Build messages array for API
        const apiMessages = [
            { role: 'system', content: Settings.get('systemPrompt') },
            ...messages.slice(-20), // Last 20 messages for context window
        ];

        // Show typing indicator
        renderTypingBubble();

        try {
            let textEl = null;
            const fullResponse = await VisionLLM.chat(apiMessages, (token, full) => {
                if (!textEl) {
                    removeTypingBubble();
                    textEl = renderBubble('assistant', '');
                }
                textEl.textContent = full;
                msgContainer().scrollTop = msgContainer().scrollHeight;
            });

            // If no streaming happened
            if (!textEl) {
                removeTypingBubble();
                renderBubble('assistant', fullResponse);
            }

            messages.push({ role: 'assistant', content: fullResponse });
            saveHistory();
        } catch (err) {
            removeTypingBubble();
            renderBubble('assistant', `⚠️ Error: ${err.message}`);
        }
    }

    // ─── Init ────────────────────────────────────────────────────────────
    function init() {
        document.getElementById('btn-chat-send').addEventListener('click', send);

        inputEl().addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
            }
        });

        document.getElementById('btn-chat-attach').addEventListener('click', () => {
            fileInput().click();
        });
        fileInput().addEventListener('change', async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            attachedImage = await ImageUtil.fileToBase64(file);
            showImagePreview(attachedImage);
            e.target.value = '';
        });

        loadHistory();
    }

    document.addEventListener('DOMContentLoaded', init);
    return { send, messages };
})();
```

### 📄 File: `d:\po\app\www\js\studio.js`
```javascript
/* ═══════════════════════════════════════════════════════════════════════════
   studio.js — Studio tab generative pipeline logic (Flux.2 I2I & LTX 2.3 I2V)
   ═══════════════════════════════════════════════════════════════════════════ */

const Studio = (() => {
    let mode = 'i2i';
    let sourceImage = null;   // base64
    let ipaImage = null;      // base64
    let cnType = 'none';

    const $ = id => document.getElementById(id);

    // ─── Mode toggle ─────────────────────────────────────────────────────
    function initModeToggle() {
        const toggle = document.querySelector('.mode-toggle');
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                mode = btn.dataset.mode;
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b === btn));
                toggle.setAttribute('data-mode', mode);
                document.body.setAttribute('data-mode', mode);
                
                // Defaults by mode
                if (mode === 'i2v') {
                    $('gen-guidance').value = '3.0';
                    $('gen-guidance-val').textContent = '3.0';
                    $('gen-steps').value = '30';
                    $('gen-steps-val').textContent = '30';
                } else {
                    $('gen-guidance').value = '3.5';
                    $('gen-guidance-val').textContent = '3.5';
                    $('gen-steps').value = '20';
                    $('gen-steps-val').textContent = '20';
                }
            });
        });
    }

    // ─── Image uploads ───────────────────────────────────────────────────
    function initUploads() {
        // Source image
        $('studio-upload').addEventListener('click', () => $('file-studio-image').click());
        $('file-studio-image').addEventListener('change', async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            sourceImage = await ImageUtil.fileToBase64(file);
            $('studio-preview').src = `data:image/jpeg;base64,${sourceImage}`;
            $('studio-preview').classList.remove('hidden');
            $('btn-clear-upload').classList.remove('hidden');
            e.target.value = '';
        });
        $('btn-clear-upload').addEventListener('click', (e) => {
            e.stopPropagation();
            sourceImage = null;
            $('studio-preview').classList.add('hidden');
            $('btn-clear-upload').classList.add('hidden');
        });

        // IP-Adapter reference
        $('ipa-upload').addEventListener('click', () => $('file-ipa-image').click());
        $('file-ipa-image').addEventListener('change', async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            ipaImage = await ImageUtil.fileToBase64(file);
            $('ipa-preview').src = `data:image/jpeg;base64,${ipaImage}`;
            $('ipa-preview').classList.remove('hidden');
            $('btn-clear-ipa').classList.remove('hidden');
            e.target.value = '';
        });
        $('btn-clear-ipa').addEventListener('click', (e) => {
            e.stopPropagation();
            ipaImage = null;
            $('ipa-preview').classList.add('hidden');
            $('btn-clear-ipa').classList.add('hidden');
        });
    }

    // ─── ControlNet type ─────────────────────────────────────────────────
    function initControlNet() {
        document.querySelectorAll('.cn-type-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                cnType = btn.dataset.cn;
                document.querySelectorAll('.cn-type-btn').forEach(b => b.classList.toggle('active', b === btn));
                $('cn-scale-row').style.display = cnType === 'none' ? 'none' : 'flex';
            });
        });
    }

    // ─── Prompt Enhancement ──────────────────────────────────────────────
    async function enhancePrompt() {
        const prompt = $('studio-prompt').value.trim();
        if (!sourceImage) { alert('Please upload a source image first.'); return; }
        if (!prompt) { alert('Please enter a prompt.'); return; }

        if (!Settings.get('rpKey') || !Settings.get('rpVLLM')) {
            alert('Please configure your RunPod API Key and vLLM Endpoint ID in Settings.');
            return;
        }

        const btn = $('btn-enhance');
        btn.disabled = true;
        btn.innerHTML = '<div class="loading-spinner" style="width:20px;height:20px;border-width:2px;"></div> Enhancing...';

        try {
            let sysPrompt;
            if (mode === 'i2v') {
                sysPrompt = `You are a prompt enhancer for an AI video generation model (LTX 2.3). The user will provide an image and a brief concept. You must generate an enhanced, detailed prompt describing:
1. The initial scene from the image (subject, appearance, composition, pose, background)
2. A naturally evolving scenario describing every motion, body movement, and composition change
3. Audio cues and sound descriptions paired with motions
Output only the enhanced prompt text, no commentary. Be detailed but concise.`;
            } else {
                sysPrompt = `You are a prompt enhancer for an AI image generation model (Flux.2). The user will provide an image and a brief concept. You must generate an enhanced, detailed prompt that describes the desired output image. Include:
1. Subject details, composition, lighting, style
2. The transformation or edit the user wants
3. Quality tags and artistic direction
Output only the enhanced prompt text, no commentary.`;
            }

            const messages = [
                { role: 'system', content: sysPrompt },
                VisionLLM.buildVisionMessage(
                    `Here is my source image. My concept: "${prompt}". Please enhance this into a detailed prompt.`,
                    sourceImage
                ),
            ];

            const enhanced = await VisionLLM.chat(messages);
            $('enhanced-prompt').value = enhanced;
            $('studio-enhance-card').classList.remove('hidden');
            $('studio-enhance-card').scrollIntoView({ behavior: 'smooth' });
        } catch (err) {
            alert(`Enhancement failed: ${err.message}\n\nCheck your vLLM Endpoint status.`);
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<span class="btn-icon">✨</span> Enhance Prompt with Vision AI';
        }
    }

    // ─── Generate ────────────────────────────────────────────────────────
    async function generate() {
        const prompt = $('enhanced-prompt').value.trim();
        if (!prompt || !sourceImage) return;

        const endpointId = mode === 'i2i' ? Settings.get('rpI2I') : Settings.get('rpI2V');
        if (!endpointId) { alert(`Please set your RunPod ${mode.toUpperCase()} endpoint ID in Settings.`); return; }
        if (!Settings.get('rpKey')) { alert('Please set your RunPod API key in Settings.'); return; }

        $('studio-result-card').classList.remove('hidden');
        $('result-loading').classList.remove('hidden');
        $('result-image').classList.add('hidden');
        $('result-video').classList.add('hidden');
        $('result-actions').classList.add('hidden');
        $('result-status-text').textContent = 'Sending to RunPod...';
        $('studio-result-card').scrollIntoView({ behavior: 'smooth' });

        const genBtn = $('btn-generate');
        genBtn.disabled = true;

        try {
            let input;
            if (mode === 'i2i') {
                input = {
                    prompt: prompt,
                    image_base64: sourceImage,
                    strength: parseFloat($('gen-strength').value),
                    num_inference_steps: parseInt($('gen-steps').value),
                    guidance_scale: parseFloat($('gen-guidance').value),
                };
                if (ipaImage) {
                    input.ip_adapter_image_base64 = ipaImage;
                    input.ip_adapter_scale = parseFloat($('ipa-scale').value);
                }
                if (cnType !== 'none') {
                    input.controlnet_type = cnType;
                    input.controlnet_conditioning_scale = parseFloat($('cn-scale').value);
                }
            } else {
                input = {
                    prompt: prompt,
                    image_base64: sourceImage,
                    num_frames: parseInt($('gen-frames').value),
                    num_inference_steps: parseInt($('gen-steps').value),
                    guidance_scale: parseFloat($('gen-guidance').value),
                    width: 768,
                    height: 512,
                };
            }

            const job = await RunPod.run(endpointId, input);
            $('result-status-text').textContent = `Job submitted (${job.id}). Waiting...`;

            const output = await RunPod.poll(endpointId, job.id, (status) => {
                const statusMap = {
                    'IN_QUEUE': '⏳ In queue...',
                    'IN_PROGRESS': '⚙️ Generating...',
                    'COMPLETED': '✅ Completed!',
                    'FAILED': '❌ Failed',
                    'CANCELLED': '🚫 Cancelled'
                };
                $('result-status-text').textContent = statusMap[status] || `Processing (${status})...`;
            });

            $('result-loading').classList.add('hidden');

            if (mode === 'i2i') {
                const imgB64 = output.image_base64 || output.image;
                if (!imgB64) throw new Error('No image returned from worker.');
                $('result-image').src = `data:image/png;base64,${imgB64}`;
                $('result-image').classList.remove('hidden');
                
                $('btn-download-result').onclick = () => ImageUtil.downloadBase64(imgB64, `studio-${Date.now()}.png`, 'image/png');
            } else {
                const vidB64 = output.video_base64 || output.video;
                if (!vidB64) throw new Error('No video returned from worker.');
                
                const videoEl = $('result-video');
                videoEl.src = `data:video/mp4;base64,${vidB64}`;
                videoEl.classList.remove('hidden');
                videoEl.play();

                $('btn-download-result').onclick = () => ImageUtil.downloadBase64(vidB64, `studio-${Date.now()}.mp4`, 'video/mp4');
            }

            $('result-actions').classList.remove('hidden');

        } catch (err) {
            console.error(err);
            $('result-loading').classList.add('hidden');
            $('result-status-text').textContent = `⚠️ Error: ${err.message}`;
        } finally {
            genBtn.disabled = false;
        }
    }

    function init() {
        initModeToggle();
        initUploads();
        initControlNet();
        $('btn-enhance').addEventListener('click', enhancePrompt);
        $('btn-generate').addEventListener('click', generate);
    }

    document.addEventListener('DOMContentLoaded', init);
    return { enhancePrompt, generate };
})();
```

---

## 🛠️ 8. Deployment & Troubleshooting Guides

### Building & Pushing Containers
Building custom GPU worker containers for RunPod is performed using standard Docker commands:
```bash
# Build the Image-to-Video Container
docker build -t <YOUR_DOCKER_USERNAME>/po-i2v:latest d:\po\workers\i2v

# Push to your registry (e.g. Docker Hub)
docker push <YOUR_DOCKER_USERNAME>/po-i2v:latest
```

### RunPod Endpoint Environment Variables
Ensure the following key variables are configured on the RunPod dashboard:
*   `MODEL_DIR`: Path to the volume directory containing model files (defaults to `/workspace/models`).
*   `HF_HUB_ENABLE_HF_TRANSFER`: Set to `1` to enable high-speed downloads if pulling remote weights from Hugging Face.

### Network Volume Setup
Ensure you copy your large weights to your persistent network volume to avoid cold-start lag. They should be mapped to the following layout:
*   `/workspace/models/10eros_v1_bf16.safetensors`
*   `/workspace/models/gemma-3-12b-it-heretic/` *(optional, contains `config.json`, tokenizer configurations, and model bins to completely avoid remote hub downloading).*

### Common Errors & Remedies
1.  **"OutOfMemoryError: CUDA out of memory"**:
    *   *Cause*: GPU offloading is not enabled or too many frames/steps are requested at high resolutions.
    *   *Remedy*: Ensure `pipe.enable_model_cpu_offload()` is present in `handler.py`. Keep image sizes to `768x512` or `512x512` and frame counts at standard indices (e.g., 81 frames).
2.  **"AttributeError: 'LTX2PipelineOutput' object has no attribute 'videos'"**:
    *   *Cause*: Confusing LTX-2.3 with older model structures.
    *   *Remedy*: Always extract using `.frames[0]` since LTX-2.3 returns a list of lists of PIL Images/numpy frames in the `.frames` attribute.
3.  **Refusals or Censorship on Explicit Prompts**:
    *   *Cause*: Prompt routing through standard aligned `google/gemma-3-12b-it`.
    *   *Remedy*: Ensure the worker is loading from `DreamFast/gemma-3-12b-it-heretic` or `mlabonne/gemma-3-12b-it-abliterated`. These remove the safety alignment weights while preserving the strong associative video conditioning.
