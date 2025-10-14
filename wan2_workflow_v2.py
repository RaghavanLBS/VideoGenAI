#!/usr/bin/env python3
"""
wan2_workflow_v2.py

Extended WAN2.2-style text->video pipeline (framework-level).

New features vs v1:
 - GPU profiling and presets (A40, rtx4090, rx series, CPU)
 - Resolution presets (480p/720p/1080p/4k) and automatic scaling
 - Control Adapter (ControlNet-like) integration (optional)
 - On-demand model load/unload with VRAM-aware decisions
 - Extra WAN params: motion_strength, noise_ratio, guidance_scale, steps
 - Feedback-based refine loop, lighting/camera/physics hints
 - Latent caching and non-destructive edits
 - CLI for fine-grained control

IMPORTANT: This is a framework. Replace WAN loader/generation/decoder calls
with vendor-provided WAN 2.2 API calls. The file includes safe placeholders
and memory-management scaffolding ready for integration.
"""

import argparse
import os
import json
import shutil
import tempfile
import random
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from gtts import gTTS
from wan2 import WAN2T2V, WANVAE, WANIPAdapter, WANControlAdapter, WANTextEncoder

import subprocess

# -------------------------
# Basic configuration
# -------------------------
BASE = Path(__file__).parent
CONFIG = {
    "assets_dir": str(BASE / "assets"),
    "autogen_dir": str(BASE / "assets" / "autogen"),
    "latents_dir": str(BASE / "cache_latents"),
    "wan_models": {
        "high": str(BASE / "models" / "wan2.2_t2v_high.safetensors"),
        "low": str(BASE / "models" / "wan2.2_t2v_low.safetensors"),
    },
    "vae_model": str(BASE / "models" / "wan_2.1_vae.safetensors"),
    "ip_adapter": str(BASE / "models" / "wan_ip_adapter.safetensors"),
    "control_adapter": str(BASE / "models" / "wan_control_adapter.safetensors"),
    "text_encoder": str(BASE / "models" / "umt5_xxl.safetensors"),
    "default_fps": 16,
    "default_frames": 64,
}

os.makedirs(CONFIG["assets_dir"], exist_ok=True)
os.makedirs(CONFIG["autogen_dir"], exist_ok=True)
os.makedirs(CONFIG["latents_dir"], exist_ok=True)

# -------------------------
# Utilities
# -------------------------

def printt(*args, **kwargs):
    print("[wan2flow]", *args, **kwargs)


def detect_gpu_profile(preferred: Optional[str] = None) -> Dict[str, Any]:
    """Detect GPU and return a small profile dict.
    preferred - allow user to force a profile like 'a40', '4090', 'cpu'
    """
    if preferred:
        name = preferred.lower()
        if "cpu" in name:
            return {"device": "cpu", "name": "cpu", "vram_gb": 0}
        if "a40" in name:
            return {"device": "cuda", "name": "A40", "vram_gb": 48}
        if "4090" in name or "rtx" in name:
            return {"device": "cuda", "name": "RTX4090", "vram_gb": 24}
        if "rx" in name:
            return {"device": "cuda", "name": "AMD_RX", "vram_gb": 24}

    if not torch.cuda.is_available():
        return {"device": "cpu", "name": "cpu", "vram_gb": 0}

    try:
        props = torch.cuda.get_device_properties(0)
        name = props.name.lower()
        total_mem = int(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
        # coarse mapping
        if "a40" in name or "a6000" in name:
            profile = {"device": "cuda", "name": "A40/A6000", "vram_gb": total_mem}
        elif "4090" in name or "rtx 4090" in name:
            profile = {"device": "cuda", "name": "RTX4090", "vram_gb": total_mem}
        else:
            profile = {"device": "cuda", "name": props.name, "vram_gb": total_mem}
        return profile
    except Exception:
        return {"device": "cuda", "name": "cuda_device", "vram_gb": 24}


RES_PRESETS = {
    "480p": {"width": 640, "height": 480},
    "720p": {"width": 1280, "height": 720},
    "1080p": {"width": 1920, "height": 1080},
    "4k": {"width": 3840, "height": 2160},
}


def get_resolution(res: str, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    if res:
        if res not in RES_PRESETS:
            raise ValueError(f"Unknown resolution preset: {res}")
        return RES_PRESETS[res]["width"], RES_PRESETS[res]["height"]
    return (width or 512, height or 512)

# -------------------------
# Identity / Embedding helpers
# -------------------------

def pil_load(path: Path, size=(512, 512)):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size, Image.LANCZOS)
    return im


def compute_embedding_from_image(path: Path, size=(128, 128)) -> torch.Tensor:
    im = pil_load(path, size=size)
    arr = np.array(im).astype(np.float32) / 255.0
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    vec = np.concatenate([mean, std])
    t = torch.from_numpy(vec).float()
    out = torch.zeros(512, dtype=torch.float32)
    out[: t.shape[0]] = t
    return out.unsqueeze(0)

# -------------------------
# WAN model handle (extended)
# -------------------------
class WANModelHandle:
    """Framework wrapper to manage WAN components and offload.

    IMPORTANT: Replace placeholder load/generate/decode methods with real WAN API calls.
    """

    def __init__(self, high_path: str, low_path: str, vae_path: str, text_encoder: Optional[str] = None, ip_adapter: Optional[str] = None, control_adapter: Optional[str] = None):
        self.high_path = high_path
        self.low_path = low_path
        self.vae_path = vae_path
        self.text_encoder = text_encoder
        self.ip_adapter = ip_adapter
        self.control_adapter = control_adapter
        self._loaded = False
        self.models = {}

    def load_models(self, device: str = "cuda", precision: str = "fp16", use_control_adapter: bool = False):
        if self._loaded:
            return
        printt(f"🚀 Loading WAN2.2 models on {device} ({precision})")

        dtype = torch.float16 if precision == "fp16" else torch.float32

        # Core model components
        self.models["high"] = WAN2T2V.from_pretrained(
            self.high_path, variant="high-noise", torch_dtype=dtype
        ).to(device)

        self.models["low"] = WAN2T2V.from_pretrained(
            self.low_path, variant="low-noise", torch_dtype=dtype
        ).to(device)

        self.models["vae"] = WANVAE.from_pretrained(self.vae_path, torch_dtype=dtype).to(device)
        self.models["text_encoder"] = WANTextEncoder.from_pretrained(self.text_encoder, torch_dtype=dtype).to(device)

        # Optional adapters
        if self.ip_adapter:
            self.models["ip_adapter"] = WANIPAdapter.from_pretrained(self.ip_adapter, torch_dtype=dtype).to(device)

        if use_control_adapter and self.control_adapter:
            self.models["control_adapter"] = WANControlAdapter.from_pretrained(self.control_adapter, torch_dtype=dtype).to(device)

        self._loaded = True
        torch.cuda.empty_cache()
        printt("✅ WAN2.2 models loaded successfully")

    def unload_models(self):
        printt("Unloading WAN models to free VRAM...")
        self.models = {}
        self._loaded = False
        torch.cuda.empty_cache()

    def generate_latent(
            self,
            prompt: str,
            prompt_embedding: Optional[torch.Tensor],
            character_embeddings: Optional[torch.Tensor],
            frames: int = 64,
            width: int = 512,
            height: int = 512,
            guidance_scale: float = 7.5,
            motion_strength: float = 1.0,
            noise_ratio: float = 0.5,
            steps: int = 20,
            seed: Optional[int] = None,
            use_control_adapter: bool = False,
            control_hint: Optional[Dict[str, Any]] = None,
        ) -> torch.Tensor:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed or 42)

            model = self.models["high"]
            cond = self.models["text_encoder"].encode(prompt)

            latent = model.generate(
                prompt_embeds=cond,
                guidance_scale=guidance_scale,
                motion_strength=motion_strength,
                noise_ratio=noise_ratio,
                num_inference_steps=steps,
                width=width,
                height=height,
                num_frames=frames,
                generator=generator,
                control_adapter=self.models.get("control_adapter") if use_control_adapter else None,
                ip_adapter=self.models.get("ip_adapter"),
            )

            return latent.cpu()


    def decode_latent_to_mp4(self, latent: torch.Tensor, out_path: str, fps: int = 16, temp_dir: Optional[str] = None):
        vae = self.models["vae"]
        printt("🎞️ Decoding latent to video frames...")

        with torch.no_grad():
            frames = vae.decode(latent.to(vae.device)).clamp(0, 1)

        frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")

        temp_dir = Path(temp_dir or tempfile.mkdtemp(prefix="wan_decode_"))
        for i, frame in enumerate(frames_np):
            Image.fromarray(frame).save(temp_dir / f"frame_{i:04d}.png")

        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(temp_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
        ], check=True)

        printt(f"✅ Video saved: {out_path}")
        shutil.rmtree(temp_dir)
        return out_path


# -------------------------
# Latent cache utilities
# -------------------------

def save_latent_to_disk(latent: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent.cpu(), path)
    printt("Saved latent:", path)


def load_latent_from_disk(path: Path) -> torch.Tensor:
    return torch.load(path)

# -------------------------
# Refinement: physics/camera/lighting
# -------------------------

def video_physics_prompt_refine(latent: torch.Tensor, prompt: str = "", strength: float = 0.35) -> torch.Tensor:
    latent = latent.clone()
    dtype = latent.dtype
    if dtype == torch.float16:
        latent = latent.float()

    p = (prompt or "").lower()

    kernel = 3
    if "smooth" in p or "gentle" in p:
        kernel = 5
    if "very smooth" in p:
        kernel = 7

    pad = kernel // 2
    latent_t = latent.unsqueeze(0) if latent.dim() == 3 else latent
    smoothed = F.avg_pool3d(latent_t.unsqueeze(0), kernel_size=(kernel, 1, 1), stride=1, padding=(pad, 0, 0)).squeeze(0)
    latent = (1 - strength) * latent + strength * smoothed

    # exposure balance
    mean_per_frame = latent.mean(dim=[1, 2, 3], keepdim=True)
    global_mean = mean_per_frame.mean()
    latent = latent + (global_mean - mean_per_frame) * 0.3

    # camera micro transforms
    if "pan left" in p:
        latent = torch.roll(latent, shifts=-2, dims=-1)
    if "pan right" in p:
        latent = torch.roll(latent, shifts=2, dims=-1)
    if "tilt up" in p:
        latent = torch.roll(latent, shifts=-2, dims=-2)
    if "tilt down" in p:
        latent = torch.roll(latent, shifts=2, dims=-2)
    if "zoom in" in p:
        latent = latent * 1.02
    if "zoom out" in p:
        latent = latent * 0.98

    # clamp
    latent = latent.clamp(-10.0, 10.0)
    if dtype == torch.float16:
        latent = latent.half()
    return latent

# -------------------------
# TTS & muxing
# -------------------------

def make_tts_wav(text: str, out_wav: str, lang: str = "en"):
    printt("Generating TTS audio (gTTS)...")
    tts = gTTS(text=text, lang=lang)
    tmp = out_wav + ".tmp.mp3"
    tts.save(tmp)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, out_wav]
    subprocess.check_call(cmd)
    os.remove(tmp)
    return out_wav


def mux_audio_video(video_mp4: str, audio_wav: str, out_path: str):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_mp4,
        "-i", audio_wav,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path
    ]
    printt("Muxing audio and video to:", out_path)
    subprocess.check_call(cmd)
    return out_path

# -------------------------
# High level flows
# -------------------------

def generate_clip(
    prompt: str,
    out_name: str,
    seed: Optional[int] = None,
    frames: Optional[int] = None,
    resolution: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    precision: str = "fp16",
    guidance_scale: float = 7.5,
    motion_strength: float = 1.0,
    noise_ratio: float = 0.5,
    steps: int = 20,
    gpu_profile: Optional[str] = None,
    use_control_adapter: bool = False,
    control_hint: Optional[Dict[str, Any]] = None,
    tts: bool = True,
):
    frames = frames or CONFIG["default_frames"]
    width, height = get_resolution(resolution, width, height)
    out_base = Path(out_name)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    profile = detect_gpu_profile(gpu_profile)
    printt("Hardware profile:", profile)

    # Basic memory-adaptive decision
    if profile["device"] == "cpu":
        precision = "fp32"

    # create or load character embeddings (demo fallback uses placeholder)
    # In production: accept --reference-image and run an identity encoder
    char_emb = torch.zeros((1, 512))

    wan = WANModelHandle(CONFIG["wan_models"]["high"], CONFIG["wan_models"]["low"], CONFIG["vae_model"], CONFIG.get("text_encoder"), CONFIG.get("ip_adapter"), CONFIG.get("control_adapter"))
    wan.load_models(device=profile["device"], precision=precision, use_control_adapter=use_control_adapter)

    latent = wan.generate_latent(
        prompt=prompt,
        prompt_embedding=None,
        character_embeddings=char_emb,
        frames=frames,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        motion_strength=motion_strength,
        noise_ratio=noise_ratio,
        steps=steps,
        seed=seed,
        use_control_adapter=use_control_adapter,
        control_hint=control_hint,
    )

    latent_path = Path(CONFIG["latents_dir"]) / f"{out_name}.latent.pt"
    save_latent_to_disk(latent, latent_path)

    # immediate auto-refine to fix obvious physics/lighting issues
    refined = video_physics_prompt_refine(latent, prompt="", strength=0.28)
    refined_path = Path(CONFIG["latents_dir"]) / f"{out_name}.refined.latent.pt"
    save_latent_to_disk(refined, refined_path)

    decode_out = str(out_base.with_suffix(".noaudio.mp4"))
    final_video = wan.decode_latent_to_mp4(refined, decode_out, fps=CONFIG["default_fps"])

    wav_out = str(out_base.with_suffix(".wav"))
    if tts:
        make_tts_wav(prompt, wav_out)

    final_out = str(out_base.with_suffix(".final.mp4"))
    if tts:
        mux_audio_video(final_video, wav_out, final_out)
    else:
        final_out = final_video

    if profile["device"] == "cuda":
        wan.unload_models()

    return {
        "latent_path": str(latent_path),
        "refined_path": str(refined_path),
        "video_noaudio": decode_out,
        "audio": wav_out if tts else None,
        "final_mp4": final_out,
    }


def refine_clip(latent_path: str, feedback_prompt: str, out_name: str, strength: float = 0.35, precision: str = "fp16"):
    latent = load_latent_from_disk(Path(latent_path))
    refined = video_physics_prompt_refine(latent, prompt=feedback_prompt, strength=strength)
    refined_path = Path(CONFIG["latents_dir"]) / f"{out_name}.refined.latent.pt"
    save_latent_to_disk(refined, refined_path)

    profile = detect_gpu_profile()
    wan = WANModelHandle(CONFIG["wan_models"]["high"], CONFIG["wan_models"]["low"], CONFIG["vae_model"], CONFIG.get("text_encoder"), CONFIG.get("ip_adapter"), CONFIG.get("control_adapter"))
    wan.load_models(device=profile["device"], precision=precision, use_control_adapter=False)

    decode_out = f"{out_name}.noaudio.mp4"
    wan.decode_latent_to_mp4(refined, decode_out, fps=CONFIG["default_fps"])

    wav_out = f"{out_name}.wav"
    make_tts_wav(feedback_prompt, wav_out)

    final_out = f"{out_name}.final.mp4"
    mux_audio_video(decode_out, wav_out, final_out)

    if profile["device"] == "cuda":
        wan.unload_models()

    return {"refined_latent": str(refined_path), "video_noaudio": decode_out, "audio": wav_out, "final_mp4": final_out}

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["generate", "refine"], required=True)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--feedback", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--latent", type=str, default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--resolution", choices=list(RES_PRESETS.keys()), default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--precision", choices=["fp32", "fp16", "fp8"], default="fp16")
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--motion", type=float, default=1.0)
    p.add_argument("--noise_ratio", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--gpu", type=str, default=None, help="Override GPU profile, e.g. a40, 4090, cpu")
    p.add_argument("--use_control_adapter", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "generate":
        res = generate_clip(
            prompt=args.prompt,
            out_name=args.out,
            seed=args.seed,
            frames=args.frames,
            resolution=args.resolution,
            width=args.width,
            height=args.height,
            precision=args.precision,
            guidance_scale=args.guidance,
            motion_strength=args.motion,
            noise_ratio=args.noise_ratio,
            steps=args.steps,
            gpu_profile=args.gpu,
            use_control_adapter=args.use_control_adapter,
        )
        printt("Generated:", json.dumps(res, indent=2))
    elif args.mode == "refine":
        if not args.latent:
            raise RuntimeError("Please provide --latent for refine mode")
        res = refine_clip(args.latent, args.feedback or args.prompt, args.out, strength=0.35, precision=args.precision)
        printt("Refined:", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
