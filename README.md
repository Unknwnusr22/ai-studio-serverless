# AI Studio

Android APK with on-device vision LLM chat + RunPod serverless i2i/i2v generation.

## Architecture

- **Chat Tab** — Talk with on-device Gemma 4 E2B vision model, attach images
- **Studio Tab** — Upload image + prompt → vision LLM enhances prompt → RunPod generates i2i or i2v
- **On-Device LLM** — GGUF model loaded from phone storage via wllama (WebAssembly llama.cpp)
- **I2I** — Flux.2 Klein (MiracleIn finetune) + IP-Adapter + ControlNet on RunPod
- **I2V** — LTX 2.3 (TenStrip/10Eros) on RunPod

## Setup

### 1. Vision Model (On-Device)

Download your GGUF model to your phone storage:
- `gemma-4-E2B-it-abliterated.Q4_K_M.gguf` (combined model + mmproj, ~1.5GB)

In the app → Settings → **Load Model** → select the .gguf file.

If you have separate model + mmproj files, select both files at once.

### 2. RunPod Setup

**Network Volume:**
1. Create a Network Volume in your preferred region
2. Upload models to the volume:
   ```
   /runpod-volume/models/
   ├── flux2-klein/          # MiracleIn Flux.2 Klein finetune
   ├── ip-adapter-flux2/     # IP-Adapter weights (optional)
   ├── controlnet-flux2/     # ControlNet weights (optional)
   └── ltx2.3-10eros/        # TenStrip LTX 2.3 10Eros
   ```

**I2I Endpoint (RTX 3090 — 24GB VRAM):**
1. Build and push Docker image:
   ```bash
   cd workers/i2i
   docker build -t yourusername/ai-studio-i2i:latest .
   docker push yourusername/ai-studio-i2i:latest
   ```
2. Create Serverless Endpoint:
   - Container image: `yourusername/ai-studio-i2i:latest`
   - GPU: RTX 3090 (24GB)
   - Attach your Network Volume
   - Set env: `MODEL_DIR=/runpod-volume/models`

**I2V Endpoint (RTX 4090 — 24GB VRAM):**
1. Build and push:
   ```bash
   cd workers/i2v
   docker build -t yourusername/ai-studio-i2v:latest .
   docker push yourusername/ai-studio-i2v:latest
   ```
2. Create Serverless Endpoint:
   - Container image: `yourusername/ai-studio-i2v:latest`
   - GPU: RTX 4090 (24GB)
   - Attach the same Network Volume
   - Set env: `MODEL_DIR=/runpod-volume/models`

### 3. App Configuration

Open the app → Settings (gear icon) and enter:
- **RunPod API Key**: Your key from runpod.io
- **I2I Endpoint ID**: From your RunPod i2i endpoint
- **I2V Endpoint ID**: From your RunPod i2v endpoint

### 4. Build APK

```bash
cd app
npm install
npx cap add android
npx cap copy android

# Build with Android Studio JDK
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\android\gradlew.bat -p android assembleDebug --no-daemon
```

The APK will be at `android/app/build/outputs/apk/debug/app-debug.apk`.

## Testing Locally

```bash
cd app
npx serve www -l 3000
```

Open `http://localhost:3000` in your browser.

## GPU Recommendations

| Endpoint | GPU | VRAM | Cost/hr (approx) |
|----------|-----|------|-------------------|
| I2I (Flux.2 Klein) | RTX 3090 | 24GB | ~$0.22 |
| I2V (LTX 2.3) | RTX 4090 | 24GB | ~$0.39 |

Both endpoints share the same Network Volume, so you only pay for storage once.
