#!/usr/bin/env bash
set -e

echo "🧠 Setting up WAN2.2 Scene JSON UI on A40..."
BASE_DIR=$(dirname "$(realpath "$0")")
cd "$BASE_DIR"

if [ ! -d "wan2env" ]; then
  python3 -m venv wan2env
fi
source wan2env/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers safetensors gtts rich moviepy streamlit tqdm pillow

MODELS_DIR="$BASE_DIR/models"
mkdir -p "$MODELS_DIR"

declare -A WAN_MODELS=(
  ["wan2.2_t2v_high.safetensors"]="https://huggingface.co/WAN-Labs/WAN2.2-T2V/resolve/main/wan2.2_t2v_high.safetensors"
  ["wan2.2_t2v_low.safetensors"]="https://huggingface.co/WAN-Labs/WAN2.2-T2V/resolve/main/wan2.2_t2v_low.safetensors"
  ["wan_2.1_vae.safetensors"]="https://huggingface.co/WAN-Labs/WAN2.1-VAE/resolve/main/vae.safetensors"
)

for model in "${!WAN_MODELS[@]}"; do
  [ -f "$MODELS_DIR/$model" ] || wget -q --show-progress -O "$MODELS_DIR/$model" "${WAN_MODELS[$model]}" || echo "⚠️ Missing $model"
done

echo "🌐 Launching Scene JSON UI..."
streamlit run wan2_ui_json.py
