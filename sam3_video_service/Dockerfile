FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv ffmpeg git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# GPU stack: CUDA 12.4 wheels + transformers>=5.0 (Sam3TrackerVideoModel).
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir \
    "transformers>=5.0.0" accelerate "huggingface_hub>=0.24"

COPY app ./app

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data /hf-cache \
    && chown -R appuser:appuser /app /data /hf-cache
USER appuser

ENV DATA_ROOT=/data \
    PORT=2129 \
    HF_HOME=/hf-cache \
    SAM3_VERSION=sam3 \
    SAM3_BACKEND=transformers \
    SAM3_MOCK=0

EXPOSE 2129

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:2129/health || exit 1

CMD ["python3", "-m", "app.main"]
