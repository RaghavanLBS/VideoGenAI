FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Prevent interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git wget ffmpeg libsndfile1 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128


# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install \
    transformers diffusers safetensors accelerate \
    pillow tqdm numpy soundfile gtts stretorch torchvision torchaudio \
    xformers flash-attn --no-build-isolation || true

# Clone ComfyUI for backend inference
#RUN git clone https://github.com/comfyanonymous/ComfyUI.git
#RUN cd ComfyUI && pip install -r requirements.txt && pip install -e .

# Copy your workflow and UI
COPY wan2_workflow_v3.py /workspace/
COPY wan2_ui_v3.py /workspace/
COPY setup.sh /workspace/

# Default command starts the Streamlit UI
CMD ["streamlit", "run", "wan2_ui_v3.py", "--server.address=0.0.0.0"]
