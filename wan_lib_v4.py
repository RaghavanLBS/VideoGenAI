"""
WAN 2.2 Workflow (no ComfyUI)
-----------------------------
Generates or decodes WAN videos using the native WAN 2.2 implementation.
"""

import os, torch, subprocess, shutil, imageio, argparse
from pathlib import Path
import numpy as np
from wan import WanT2V  # make sure WAN 2.2 repo is on PYTHONPATH

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def printt(*args):
    print("[wan2flow]", *args, flush=True)

def load_latent(path):
    printt(f"Loading latent tensor from {path}")
    latent = torch.load(path, map_location="cpu")
    if isinstance(latent, dict) and "latent" in latent:
        latent = latent["latent"]
    printt("Latent shape:", tuple(latent.shape))
    return latent

# -----------------------------------------------------------------------------
# WAN 2.2 Engine Wrapper
# -----------------------------------------------------------------------------

class WANEngine:
    def __init__(self, ckpt_dir="models", device="cuda", precision="fp16"):
        self.device = device
        self.precision = precision
        self.ckpt_dir = ckpt_dir
        self.wan_t2v = None

    def load_models(self):
        printt("Loading WAN 2.2 model from", self.ckpt_dir)
        from omegaconf import OmegaConf
        def ensure_config(cfg):
            if isinstance(cfg, str):
                return OmegaConf.load(cfg)
            return cfg

        config = ensure_config(config_path)
        self.wan_t2v = WanT2V(config, checkpoint_dir)
        printt("✅ WAN 2.2 model loaded")

    def generate_latent(self, prompt, frames=32, size=(512,512), seed=42):
        printt(f"Generating {frames} frames for prompt:", prompt)
        video = self.wan_t2v.generate(
            prompt,
            frame_num=frames,
            size=size,
            seed=seed,
            return_latent=True
        )
        latent = video["latent"]
        printt("✅ Latent generated:", tuple(latent.shape))
        return latent

    def decode_latent_to_mp4(self, latent, out_path, fps=16, keep_frames=True):
        printt("[WAN] Decoding latent → frames → MP4 …")
        vae = self.wan_t2v.vae  # native WAN 2.2 VAE
        latent = latent.to(self.device)

        T = latent.shape[2] if latent.dim() == 5 else 1
        if T == 0:
            raise RuntimeError("Empty latent tensor (T=0).")

        scene_name = Path(out_path).stem
        frames_dir = Path("frames") / scene_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        printt(f"[WAN] Saving frames to {frames_dir}")

        saved = 0
        for i in range(T):
            z = latent[:, :, i, :, :].to(self.device)
            with torch.no_grad():
                recon = vae.decode(z)
            recon = (recon.clamp(-1,1) + 1) / 2
            frame = (recon[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(frames_dir / f"frame_{i:04d}.png", frame)
            saved += 1

        printt(f"[WAN] ✅ Saved {saved} frames, invoking ffmpeg …")
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)
        ])
        printt(f"[WAN] ✅ MP4 written: {out_path}")

        if not keep_frames:
            shutil.rmtree(frames_dir)
            printt("[WAN] Frames deleted")

# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate","decode"], required=True)
    parser.add_argument("--prompt", type=str, default="A fox walking in the forest")
    parser.add_argument("--latent", type=str)
    parser.add_argument("--out", type=str, default="outputs/out.mp4")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--size", type=int, nargs=2, default=[512,512])
    args = parser.parse_args()

    engine = WANEngine()
    engine.load_models()

    if args.mode == "generate":
        latent = engine.generate_latent(args.prompt, frames=args.frames, size=tuple(args.size))
        torch.save(latent, "cache_latents/generated.latent.pt")
        engine.decode_latent_to_mp4(latent, args.out, fps=args.fps)
    elif args.mode == "decode":
        latent = load_latent(args.latent)
        engine.decode_latent_to_mp4(latent, args.out, fps=args.fps)

if __name__ == "__main__":
    main()
