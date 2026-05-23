FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and torchvision nightly with native Blackwell sm_120 support
RUN pip uninstall -y torch torchvision && \
    pip install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128


# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Download LTX-2.3 Diffusers config (keeps pristine audio-visual config for 10Eros joint inference)
RUN hf download diffusers/LTX-2.3-Diffusers --include "*.json" --local-dir /app/config

# RunPod network volume mount point
VOLUME /runpod-volume

CMD ["python", "-u", "handler.py"]
