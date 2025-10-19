#!/usr/bin/env python3
"""
WAN2.2 universal runner (task-based, config-free)

Usage example:
python wan_lib_v4.py \
  --task t2v-A14B \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --size 1280*720 \
  --offload_model True \
  --convert_model_dtype
"""

import argparse
import logging
import os
import sys
import traceback
import torch

# ---------------------------------------------------------------------------
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("WAN-Lib")

# ---------------------------------------------------------------------------
def add_repo_to_path():
    """Auto-detect and add WAN2.2 repo path."""
    candidates = [
        os.getcwd(),
        os.path.join(os.getcwd(), "Wan2.2"),
        "/workspace/Wan2.2",
    ]
    for p in candidates:
        if os.path.isdir(p):
            if p not in sys.path:
                sys.path.insert(0, p)
                logger.info(f"Added repo path: {p}")
                return p
    logger.warning("Could not auto-locate WAN2.2 repo.")
    return None


def check_flash_attention(disable_flash=False):
    """Ensure Flash-Attention is within supported range or disabled."""
    if disable_flash:
        os.environ["WAN_DISABLE_FLASH_ATTN"] = "1"
        logger.warning("FlashAttention manually disabled.")
        return
    try:
        import flash_attn
        from packaging import version
        v = version.parse(flash_attn.__version__)
        if not (version.parse("2.7.1") <= v <= version.parse("2.8.2")):
            logger.warning(f"Incompatible FlashAttention version {v}; disabling.")
            os.environ["WAN_DISABLE_FLASH_ATTN"] = "1"
    except Exception:
        os.environ["WAN_DISABLE_FLASH_ATTN"] = "1"
        logger.info("FlashAttention not found — using fallback attention.")


def list_available_tasks():
    """Return list of supported task names from wan.config if possible."""
    try:
        from wan.configs import get_config
        import inspect
        src = inspect.getsource(get_config)
        tasks = []
        for line in src.splitlines():
            if "if task ==" in line or "elif task ==" in line:
                t = line.split("==")[-1].strip().strip('"\'').strip(":")
                if t:
                    tasks.append(t)
        return tasks
    except Exception:
        return []


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="WAN2.2 Task Runner (t2v)")
    parser.add_argument("--task", required=True, help="Model task name (e.g. t2v-A14B)")
    parser.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate video from")
    parser.add_argument("--size", default="1280*720", help="Resolution, e.g. 1280*720")
    parser.add_argument("--offload_model", action="store_true", help="Offload models to CPU during sampling")
    parser.add_argument("--convert_model_dtype", action="store_true", help="Convert model params to config.param_dtype")
    parser.add_argument("--disable_flash", action="store_true", help="Disable FlashAttention completely")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--out_dir", default="./outputs", help="Directory to save outputs")
    parser.add_argument("--list_tasks", action="store_true", help="List available --task options and exit")
    args = parser.parse_args()

    if args.list_tasks:
        tasks = list_available_tasks()
        if tasks:
            print("Available WAN tasks:")
            for t in tasks:
                print("  ", t)
        else:
            print("Could not auto-detect tasks — open wan/config/__init__.py to inspect.")
        sys.exit(0)

    # --- Environment setup -------------------------------------------------
    add_repo_to_path()
    check_flash_attention(args.disable_flash)

    # --- Import WAN modules dynamically -----------------------------------
    try:
        from wan.text2video import WanT2V
        from wan.config import get_config
    except ImportError as e:
        logger.error("Could not import from WAN2.2 repo. Make sure 'Wan2.2/' is cloned and accessible.")
        logger.error(e)
        sys.exit(1)

    # --- Load task-based config -------------------------------------------
    config = get_config(task=args.task)
    logger.info(f"Loaded config for task {args.task}")

    # --- Parse resolution --------------------------------------------------
    try:
        if "*" in args.size:
            w, h = map(int, args.size.split("*"))
        elif "x" in args.size.lower():
            w, h = map(int, args.size.lower().split("x"))
        else:
            w, h = 1280, 720
        size = (w, h)
    except Exception:
        size = (1280, 720)
        logger.warning(f"Invalid size format {args.size}, defaulting to {size}")

    # --- Initialize WAN model ---------------------------------------------
    model = WanT2V(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device,
        convert_model_dtype=args.convert_model_dtype,
    )

    # --- Generate video ----------------------------------------------------
    logger.info(f"Generating video for prompt: {args.prompt!r}")
    videos = model.generate(
        input_prompt=args.prompt,
        size=size,
        frame_num=args.frames,
        sampling_steps=args.steps,
        offload_model=args.offload_model,
    )

    # --- Save output -------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_tensor = os.path.join(args.out_dir, "wan_output.pt")
    torch.save(videos.cpu(), out_tensor)
    logger.info(f"✅ Saved output tensor to {out_tensor}")

    # Optionally save a quick preview image
    try:
        from PIL import Image
        frame = videos[0].detach().cpu()
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frame = (frame * 255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(frame).save(os.path.join(args.out_dir, "preview.png"))
        logger.info("Saved preview.png")
    except Exception:
        pass


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal error:")
        logger.error(traceback.format_exc())
        sys.exit(1)
