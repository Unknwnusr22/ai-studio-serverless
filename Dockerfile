FROM runpod/worker-v1-vllm:v2.18.1

RUN pip install "transformers>=4.57.0" --upgrade --no-cache-dir

ENV MODEL_NAME=/runpod-volume/models/qwen2-vl-8b-fp8-abliterated
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV MAX_MODEL_LEN=8192
ENV TENSOR_PARALLEL_SIZE=1
ENV DTYPE=bfloat16
ENV DISABLE_LOG_REQUESTS=1
