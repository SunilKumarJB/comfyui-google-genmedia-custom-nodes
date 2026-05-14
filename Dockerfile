# ComfyUI on Cloud Run with Google GenMedia custom nodes
# These custom nodes call Vertex AI APIs, so heavy generation runs on
# Google's infra — ComfyUI here only orchestrates, so CPU-only is fine.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    COMFYUI_PATH=/app/ComfyUI

# System deps: git for cloning, ffmpeg/libGL for moviepy + opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Pin ComfyUI to a known-good release ---
# The repo moved to Comfy-Org/ComfyUI; the old comfyanonymous URL still
# redirects but the canonical one is preferred. Update this tag periodically.
ARG COMFYUI_REF=v0.3.65
RUN git clone --depth 1 --branch ${COMFYUI_REF} \
        https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH}

# Install CPU-only torch first (the default torch wheel pulls in 2GB+ of
# CUDA libs we don't need, since the GenMedia nodes call Vertex AI rather
# than running models locally).
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio \
    && pip install -r ${COMFYUI_PATH}/requirements.txt

# --- Install Google GenMedia custom nodes ---
ARG CUSTOM_NODES_REPO=https://github.com/SunilKumarJB/comfyui-google-genmedia-custom-nodes
ARG CUSTOM_NODES_REF=main
RUN git clone --depth 1 --branch ${CUSTOM_NODES_REF} \
        ${CUSTOM_NODES_REPO} \
        ${COMFYUI_PATH}/custom_nodes/comfyui-google-genmedia-custom-nodes \
    && pip install -r ${COMFYUI_PATH}/custom_nodes/comfyui-google-genmedia-custom-nodes/requirements.txt

# --- Optional helper custom nodes used by sample workflows ---
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
        ${COMFYUI_PATH}/custom_nodes/ComfyUI-VideoHelperSuite \
    && git clone --depth 1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts \
        ${COMFYUI_PATH}/custom_nodes/ComfyUI-Custom-Scripts

# Move sample workflows into ComfyUI's workflows dir and seed input images.
# Using `find -exec` instead of glob+`||true` so real errors aren't silently
# swallowed.
RUN mkdir -p ${COMFYUI_PATH}/user/default/workflows ${COMFYUI_PATH}/input \
    && SAMPLE_DIR=${COMFYUI_PATH}/custom_nodes/comfyui-google-genmedia-custom-nodes/sample-workflows \
    && if [ -d "$SAMPLE_DIR" ]; then \
         find "$SAMPLE_DIR" -maxdepth 1 -name '*.json' -exec cp {} ${COMFYUI_PATH}/user/default/workflows/ \; ; \
         if [ -d "$SAMPLE_DIR/input-images" ]; then \
           cp -r "$SAMPLE_DIR/input-images/." ${COMFYUI_PATH}/input/ ; \
         fi ; \
       fi

WORKDIR ${COMFYUI_PATH}

# Cloud Run requires the container to listen on $PORT (default 8080).
# We pass it through to ComfyUI's --port flag.
ENV PORT=8080
EXPOSE 8080

# Notes on the CMD:
#  --listen 0.0.0.0  → bind all interfaces (Cloud Run requirement)
#  --port ${PORT}    → bind to whatever Cloud Run injected
#  --cpu             → ComfyUI flag; force CPU-only mode
#  shell form (no JSON array) is intentional so $PORT expands at runtime.
#  `exec` replaces the shell with python so SIGTERM reaches the process
#  on instance shutdown.
CMD exec python main.py \
        --listen 0.0.0.0 \
        --port ${PORT} \
        --cpu
