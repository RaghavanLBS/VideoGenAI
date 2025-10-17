# =====================================================
# WAN2.2 A40 Docker Image (Lightweight + Mount Models)
# =====================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# -----------------------------------------------------
# System packages
# -----------------------------------------------------
RUN apt-get update && apt-get install -y ffmpeg git wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# Working directory
# -----------------------------------------------------
WORKDIR /app
COPY . /app

# -----------------------------------------------------
# Python dependencies
# -----------------------------------------------------
RUN pip install --upgrade pip && \
    pip install diffusers transformers safetensors gtts moviepy streamlit tqdm pillow rich

# -----------------------------------------------------
# Environment configuration
# -----------------------------------------------------
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# -----------------------------------------------------
# Expose Streamlit
# -----------------------------------------------------
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "wan2_ui_json.py"]
