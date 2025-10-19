#!/usr/bin/env python3
"""
wan_lib_v4.py — Robust WAN2.2 loader + runner

Features:
- Adds repo path to PYTHONPATH (auto-detect)
- Loads config via OmegaConf
- Detects FlashAttention and auto-disables if unavailable or wrong version
- Safe model initialization with WanT2V
- CLI arguments for quick testing/generation
"""

import argparse
import importlib
import logging
import os
import sys
import traceback
from typing import Any

# --- Logging ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("wan_lib_v4")

# --- Helpers ---------------------------------------------------------------
def add_repo_to_path(repo_names=("Wan2.2", "Wan2", "wan2", "wan")):
    """
    Try to find the local WAN repo directory in a few likely places and add it to sys.path.
    Looks in cwd, parent, /workspace, and environment variable WAN2_ROOT.
    Returns path added or None.
    """
    # if environment variable provided, prefer it
    env_root = os.environ.get("WAN2_ROOT")
    candidates = []
    if env_root:
        candidates.append(env_root)
    cwd = os.getcwd()
    candidates += [
        cwd,
        os.path.join(cwd, "Wan2.2"),
        os.path.join(cwd, "Wan2"),
        os.path.join(cwd, "wan"),
        "/workspace/Wan2.2",
        "/workspace/Wan2",
        "/workspace/wan",
        os.path.dirname(cwd),  # parent
    ]
    # dedupe
    seen = set()
    added = None
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        for rname in repo_names:
            p = os.path.join(c) if os.path.basename(c).lower().startswith(rname.lower()) else os.path.join(c, rname)
            if os.path.isdir(p):
                if p not in sys.path:
                    sys.path.insert(0, p)
                    logger.info(f"Added WAN repo to sys.path: {p}")
                    added = p
                return added
    # fallback: try CWD itself
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
        logger.info(f"Added CWD to sys.path: {cwd}")
        return cwd
    return added

def ensure_config(config_or_path: Any):
    """
    If passed a string, assume it's a path to a YAML config and load it with OmegaConf.
    If passed a dict-like or OmegaConf already, return a usable object.
    """
    try:
        from omegaconf import OmegaConf
    except Exception as e:
        logger.error("OmegaConf is required but not installed. Install with `pip install omegaconf`.")
        raise

    if isinstance(config_or_path, str):
        if not os.path.exists(config_or_path):
            raise FileNotFoundError(f"Config file not found: {config_or_path}")
        cfg = OmegaConf.load(config_or_path)
        logger.info(f"Loaded config from {config_or_path}")
        return cfg
    # If dict, convert to OmegaConf
    try:
        # OmegaConf.create will pass through OmegaConf objects
        cfg = OmegaConf.create(config_or_path)
        return cfg
    except Exception as e:
        # If it's already a simple object with attributes, return as-is
        return config_or_path

def check_and_handle_flash_attn(disable_flag: bool):
    """
    Try to import flash_attn and ensure version is within allowed range (>=2.7.1 <=2.8.2).
    If disable_flag is set or version mismatch/unavailable, set environment var to disable and return False.
    Otherwise return True.
    """
    if disable_flag:
        os.environ["WAN_DISABLE_FLASH_ATTN"] = "1"
        logger.warning("FlashAttention explicitly disabled via CLI flag.")
        return False

    try:
        import importlib
        flash = importlib.import_module("flash_attn")
        v = getattr(flash, "__version__", None) or getattr(flash, "version", None)
        logger.info(f"Detected flash-attn version: {v}")
        # version comparison: accept 2.7.1 - 2.8.2 inclusive
        def parse_ver(x):
            return tuple(int(p) for p in str(x).split("+")[0].split(".") if p.isdigit())
        try:
            ver_tuple = parse_ver(v)
            if (ver_tuple >= (2, 7, 1)) and (ver_tuple <= (2, 8, 2)):
                logger.info("flash-attn version within supported range.")
                return True
            else:
                logger.warning(
                    "flash-attn version outside supported range (>=2.7.1 and <=2.8.2). Disabling flash-attn fallback."
                )
                os.environ["WAN_DISABLE_FLASH_ATTN"] = "1"
                return False
        except Exception:
            logger.warning("Could not parse flash-attn version string. Disabling to be safe.")
            os.environ["WAN_DISABLE_FLASH_ATTN"] = "1"
            return False
    except Exception as e:
        logger.info("flash-attn not available; will use PyTorch/xformers fallback if available.")
        os.environ.setdefault("WAN_DISABLE_FLASH_ATTN", "1")
        return False

def import_wan_t2v():
    """
    Try importing WanT2V from common module paths used in forks.
    Returns the class if found, else raises ImportError with helpful message.
    """
    tried = []
    possible_paths = [
        ("Wan2.2.wan.text2video", "WanT2V"),
        ("wan.text2video", "WanT2V"),
        ("Wan2.wan.text2video", "WanT2V"),
        ("wan2.wan.text2video", "WanT2V"),
        ("wan.text2video", "WanT2V"),
        ("text2video", "WanT2V"),  # local file
    ]
    for mod_path, cls_name in possible_paths:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            logger.info(f"Imported {cls_name} from {mod_path}")
            return cls
        except Exception as exc:
            tried.append((mod_path, exc))
    # nothing found
    msg = "Could not import WanT2V from known locations. Tried:\n"
    for m, e in tried:
        msg += f" - {m}: {type(e).__name__}: {e}\n"
    raise ImportError(msg)

# --- Engine ---------------------------------------------------------------
class Engine:
    def __init__(self, config, checkpoint_dir, device_id=0, disable_flash=False):
        self.repo_path = add_repo_to_path()
        self.disable_flash = disable_flash
        check_and_handle_flash_attn(disable_flag=disable_flash)
        self.config = ensure_config(config)
        self.checkpoint_dir = checkpoint_dir
        self.device_id = int(device_id)
        self.wan_t2v = None

    def load_models(self):
        """
        Instantiate WanT2V with safe arguments. Handles string/dict config input.
        """
        WanT2V = import_wan_t2v()
        # WanT2V signature: WanT2V(config, checkpoint_dir, device_id=0, rank=0, ...)
        try:
            # Some repos expect checkpoint_dir first or named kwargs; use named when possible.
            # Ensure config is OmegaConf/dict-like with attributes expected in constructor.
            self.wan_t2v = WanT2V(config=self.config, checkpoint_dir=self.checkpoint_dir, device_id=self.device_id)
            logger.info("WanT2V initialized successfully.")
        except TypeError as e:
            # Fallback: positional ordering (config, checkpoint_dir, device)
            logger.warning(f"TypeError on WanT2V init with kwargs: {e}. Trying positional args.")
            try:
                self.wan_t2v = WanT2V(self.config, self.checkpoint_dir, self.device_id)
                logger.info("WanT2V initialized using positional args.")
            except Exception as e2:
                logger.error("Failed to initialize WanT2V using both kw and positional args.")
                raise

    def generate(self, prompt: str, out_path: str = None, **generate_kwargs):
        """
        Simple wrapper to call WanT2V.generate(...) and optionally save output.
        Returns the generated tensor or path to saved file if `out_path` is set and rank==0.
        """
        if self.wan_t2v is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        # default generation parameters can be overridden via generate_kwargs
        params = dict(
            input_prompt=prompt,
            size=(1280, 720),
            frame_num=81,
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
        )
        params.update(generate_kwargs)
        logger.info(f"Starting generation with prompt: {prompt!r} and params: {params}")
        videos = self.wan_t2v.generate(**params)
        # videos likely a tensor or None (if not rank 0). Save if requested and rank==0
        if out_path and videos is not None:
            # attempt save using torchvision/ffmpeg if available, else save raw tensor
            try:
                import torchvision
                import torchvision.transforms as T
                import numpy as np
                # videos shape typically [C, N*H, W] per the repo commentary. Adjust saving according to shape.
                # We'll attempt a minimal save: save first frame as PNG and the whole tensor as .pt
                if isinstance(videos, torch.Tensor):
                    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                    # save full tensor
                    torch.save(videos.cpu(), out_path + ".pt")
                    logger.info(f"Saved tensor to {out_path}.pt")
                    # Try saving the center frame as image for quick inspect
                    try:
                        C = videos.shape[0]
                        if C >= 3:
                            # attempt to extract center frame — assume frames stacked along second axis
                            # This is heuristic; user can modify as needed.
                            N = (videos.shape[1])  # possibly N*H depending on vae decode; can't be sure
                            frame = videos[:, 0, :, :].cpu() if videos.dim() == 4 else videos[:, 0, :, :].cpu()
                            # clamp and convert
                            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                            frame = (frame * 255).byte().permute(1, 2, 0).numpy()
                            from PIL import Image
                            im = Image.fromarray(frame)
                            im.save(out_path + "_frame0.png")
                            logger.info(f"Saved preview frame to {out_path}_frame0.png")
                    except Exception as e:
                        logger.debug(f"Couldn't save preview frame: {e}")
            except Exception as e:
                logger.warning(f"Could not save video using torchvision (falling back to .pt): {e}")
                try:
                    torch.save(videos.cpu(), out_path + ".pt")
                    logger.info(f"Saved tensor to {out_path}.pt")
                except Exception as e2:
                    logger.error(f"Failed to save output: {e2}")
        return videos

# --- CLI -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="WAN2.2 lightweight runner (wan_lib_v4 replacement).")
    p.add_argument("--config", "-c", required=True, help="Path to WAN2.2 config YAML (or provide imported object path).")
    p.add_argument("--ckpt", "-k", required=True, help="Path to checkpoint dir (WAN2.2 checkpoints).")
    p.add_argument("--device", "-d", default=0, help="CUDA device id (default 0).")
    p.add_argument("--disable-flash", action="store_true", help="Disable flash-attn even if present.")
    p.add_argument("--prompt", "-p", default=None, help="If set, will run a single generation with this prompt.")
    p.add_argument("--out", "-o", default="./wan_out", help="Base path to save outputs (no extension).")
    p.add_argument("--frames", type=int, default=81, help="Number of frames to generate (default 81).")
    p.add_argument("--steps", type=int, default=50, help="Sampling steps.")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        logger.info("Starting WAN loader...")
        # Add repo to path
        add_repo_to_path()
        # Create engine
        engine = Engine(config=args.config, checkpoint_dir=args.ckpt, device_id=args.device, disable_flash=args.disable_flash)
        engine.load_models()
        if args.prompt:
            out_path = args.out
            videos = engine.generate(
                prompt=args.prompt,
                out_path=out_path,
                frame_num=args.frames,
                sampling_steps=args.steps
            )
            logger.info("Generation done.")
        else:
            logger.info("No prompt provided. Models loaded and ready.")
    except Exception as e:
        logger.error("Fatal error in wan_lib_v4:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
