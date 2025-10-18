#!/usr/bin/env bash
# setup.sh — minimal install for WAN2 Workflow V3 (gTTS only)
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "==> Installing system dependencies..."
sudo apt update -y
sudo apt install -y python3-venv python3-pip ffmpeg git wget build-essential libsndfile1

if [ ! -d "wan2env" ]; then
  echo "==> Creating virtual environment wan2env"
  python3 -m venv wan2env
fi
source wan2env/bin/activate

echo "==> Upgrading pip and installing Python packages"
pip install --upgrade pip setuptools wheel

# ✅ Torch and essentials
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers diffusers safetensors accelerate ftfy
pip install pillow tqdm numpy soundfile gtts streamlit
pip install xformers flash-attn --no-build-isolation

 

# ✅ Optional: ComfyUI (for model loader later, no heavy dependencies)
if [ ! -d "ComfyUI" ]; then
  echo "==> Cloning ComfyUI (for model loaders, optional)"
  git clone https://github.com/comfyanonymous/ComfyUI.git
fi
cd ComfyUI
pip install -r requirements.txt || true
pip install -e .
cd "$BASE_DIR"

# ✅ Model downloads (Comfy-Org WAN2.2)
mkdir -p models
declare -A WAN_MODELS=(
  ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
  ["wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
  ["wan_2.1_vae.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
  ["umt5_xxl_fp16.safetensors"]="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors"
)

echo "==> Downloading model files if missing..."
for name in "${!WAN_MODELS[@]}"; do
  url="${WAN_MODELS[$name]}"
  if [ ! -f "models/$name" ]; then
    echo "Downloading $name..."
    wget --show-progress -c "$url" -O "models/$name" || echo "⚠️ Failed to download $name (please download manually if needed)"
  else
    echo "Exists: models/$name"
  fi
done

echo "==> Setup complete!"
echo "Activate the venv: source wan2env/bin/activate"
echo "Start UI: streamlit run wan2_ui_v3.py"
