#!/usr/bin/env python3
# wan2_workflow_v3.py
# (ComfyUI-aware WANEngine + full pipeline - updated)
# Save this file as wan2_workflow_v3.py

from __future__ import annotations
import argparse, json, os, random, shutil, subprocess, tempfile, time, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch, imageio  
from PIL import Image
 
import torch.nn.functional as F
# at top of wan2_workflow_v3.py
import sys
sys.path.append("ComfyUI")  # so comfy modules can be imported

import comfy.model_management as mm
import comfy.sample as comfy_sample
import comfy.model_management as mm
import comfy.utils as comfy_utils
 

from comfy.diffusers_load import load_diffusers
from comfy.sd import load_checkpoint_guess_config


# CLIP for IPAdapter
from transformers import CLIPImageProcessor, CLIPVisionModel,AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# cached tokenizer
_TOKENIZER = None
def get_tokenizer(name="google/umt5-xxl"):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER
# optional libs
try:
    import soundfile as sf
    SOUND_FILE_AVAILABLE = True
except Exception:
    SOUND_FILE_AVAILABLE = False

# try ComfyUI import (if installed in python env or cloned into project)
COMFY_AVAILABLE = False
COMFY = None
try:
    import comfy
    COMFY_AVAILABLE = True
    COMFY = comfy
except Exception:
    # try if ComfyUI is in sibling folder ./ComfyUI
    comfy_path = Path(__file__).parent / "ComfyUI"
    if comfy_path.exists():
        sys.path.append(str(comfy_path))
        try:
            import comfy
            COMFY_AVAILABLE = True
            COMFY = comfy
        except Exception:
            COMFY_AVAILABLE = False

# ---- basic paths / config ----
BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
CACHE_DIR = BASE / "cache_latents"
OUTPUTS_DIR = BASE / "outputs"
PERSONA_CACHE = BASE / "persona_cache"
ASSETS_DIR = BASE / "assets"

for p in (MODELS_DIR, CACHE_DIR, OUTPUTS_DIR, PERSONA_CACHE, ASSETS_DIR):
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    "fp_precision": "fp16",
    "default_fps": 16,
    "default_frames": 64,
    "default_width": 512,
    "default_height": 512,
}

RES_PRESETS = {
    "480p": {"width": 640, "height": 480},
    "720p": {"width": 1280, "height": 720},
    "1080p": {"width": 1920, "height": 1080},
    "4k": {"width": 3840, "height": 2160},
}

'''
DEFAULT_MODELS = {
    "unet_high": MODELS_DIR / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "unet_low": MODELS_DIR / "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    "vae": MODELS_DIR / "wan_2.1_vae.safetensors",
    "text_encoder": MODELS_DIR / "umt5_xxl_fp16.safetensors",
    # ip_adapter optional (we will not require it)
    "ip_adapter": MODELS_DIR / "wan_ip_adapter.safetensors",
}
from pathlib import Path
'''

DEFAULT_MODELS = {
    "unet_high": Path("models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
    "unet_low":  Path("models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
    "vae":       Path("models/wan_2.1_vae.safetensors"),
    "text_encoder": Path("models/umt5_xxl_fp16.safetensors")
}

# ---- utilities ----

def set_random_seed(seed=None):
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[WAN] Using seed: {seed}")
    return seed

def printt(*args, **kwargs):
    print("[wan2flow]", *args, **kwargs)

def detect_gpu_profile(preferred: Optional[str] = None) -> Dict[str, Any]:
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
        if "a40" in name or "a6000" in name:
            profile = {"device": "cuda", "name": "A40/A6000", "vram_gb": total_mem}
        elif "4090" in name or "rtx 4090" in name:
            profile = {"device": "cuda", "name": "RTX4090", "vram_gb": total_mem}
        else:
            profile = {"device": "cuda", "name": props.name, "vram_gb": total_mem}
        return profile
    except Exception:
        return {"device": "cuda", "name": "cuda_device", "vram_gb": 24}

def get_resolution(res: Optional[str], width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    if res:
        if res not in RES_PRESETS:
            raise ValueError(f"Unknown resolution preset: {res}")
        return RES_PRESETS[res]["width"], RES_PRESETS[res]["height"]
    return (width or DEFAULT_CONFIG["default_width"], height or DEFAULT_CONFIG["default_height"])

# ---- IPAdapterCLIP ----
class IPAdapterCLIP:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.processor = None
    def _ensure(self):
        if self.model is None:
            printt("Loading CLIP vision model for IPAdapter")
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.model.eval()
    def compute_embedding(self, image_path: str) -> torch.Tensor:
        self._ensure()
        img = Image.open(image_path).convert("RGB")
        proc = self.processor(images=img, return_tensors="pt")
        for k,v in proc.items():
            proc[k] = v.to(self.device)
        with torch.no_grad():
            out = self.model(**proc).pooler_output
        emb = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return emb.detach().cpu()
    def batch_compute(self, image_paths: List[str]) -> torch.Tensor:
        embs = [self.compute_embedding(p) for p in image_paths]
        return torch.cat(embs, dim=0)
    def save_persona(self, name: str, emb: torch.Tensor):
        PERSONA_CACHE.mkdir(parents=True, exist_ok=True)
        path = PERSONA_CACHE / f"{name}.pt"
        torch.save(emb, path)
        printt(f"Saved persona embedding -> {path}")
    def load_persona(self, name: str) -> Optional[torch.Tensor]:
        path = PERSONA_CACHE / f"{name}.pt"
        if path.exists():
            return torch.load(path)
        return None

# ---- WANEngine (ComfyUI-aware) ----
@dataclass
class WANEngine:
    models_dir: Path = MODELS_DIR
    device: str = "cuda"
    precision: str = "fp16"
    low_vram: bool = False
    models: Dict[str, Any] = None
    comfy_loaded: bool = False

    def __init__(self, models_dir="models", device ="cuda", precision="fp16"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.precision = precision
        self.pipe = None

    def __post_init__(self):
        self.models = {}
        self.comfy_loaded = False
        # if ComfyUI is available, try to wire its loader functions
        if COMFY_AVAILABLE:
            try:
                # we'll try to import comfy.model_management or comfy.sd loader functions
                # Comfy's internal API varies across versions — we attempt common entrypoints
                import importlib
                self.comfy = importlib.import_module("comfy.model_management") if importlib.util.find_spec("comfy.model_management") else importlib.import_module("comfy.sd")
                self.comfy_loaded = True
                printt("ComfyUI detected: WANEngine will try to use Comfy model loaders if invoked.")
            except Exception:
                # fallback: Comfy is present but APIs may differ
                self.comfy_loaded = False
        else:
            self.comfy_loaded = False

    def load_models(self, device="cuda", high: Optional[Path] = None, low: Optional[Path] = None, vae: Optional[Path] = None, text_encoder: Optional[Path] = None, use_ip_adapter: bool = False):
        print("[WAN] Loading UNet, VAE, and Text Encoder...")
        from safetensors.torch import load_file

        unet = comfy.sd.load_diffusion_model("models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors")
        vae_state = load_file("models/wan_2.1_vae.safetensors")

        from comfy.sd import load_checkpoint_guess_config
        vae = load_checkpoint_guess_config("models/wan2_vae.safetensors", output_vae=True)

        from safetensors.torch import load_file
        from transformers import T5EncoderModel, T5Config
        
        clip_weights = load_file("models/umt5_xxl_fp16.safetensors")

        # Initialize a matching model config
        config = T5Config.from_pretrained("google/umt5-xxl", torch_dtype="float16")
        clip = T5EncoderModel(config)

        # Load weights
        missing, unexpected = clip.load_state_dict(clip_weights, strict=False)
        print(f"[WAN] CLIP/T5 loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")

        clip = clip.to("cuda", dtype=torch.float16)

        self.pipe = {"unet": unet, "vae": vae, "clip": clip}
        print(f"[WAN] ✅ Models loaded successfully ")
         
        

    def _register_placeholder_paths(self, high, low, vae, text_encoder):
        printt("WANEngine: (placeholder) registering model paths")
        self.models['unet_high'] = str(high) if high else None
        self.models['unet_low'] = str(low) if low else None
        self.models['vae'] = str(vae) if vae else None
        self.models['text_encoder'] = str(text_encoder) if text_encoder else None
        self.models['ip_adapter'] = str(self.models_dir / 'wan_ip_adapter.safetensors') if (self.models_dir / 'wan_ip_adapter.safetensors').exists() else None
        printt("WAN models registered (placeholders). Replace with real loader in WANEngine.load_models().")

    def unload_models(self):
        printt("WANEngine: unloading placeholder models")
        self.models = {}
        torch.cuda.empty_cache()


    


    def sample_latent(self, prompt, frames=32, width=1280, height=720,
                      guidance=7.5, steps=10, seed=None, prev_latent_context:Optional[torch.Tensor] = None):
   
        
        # device/dtype
        device = mm.get_torch_device()
        dtype = torch.float16

        # set seed (use your set_random_seed helper)
        seed = set_random_seed(seed)
        print(f"[WAN] Sampling latent (seed={seed}) …")

        unet = self.pipe["unet"]
        clip = self.pipe["clip"]
        vae = self.pipe.get("vae", None)

        # --- 1) Tokenize and encode prompt ---
        tokenizer = get_tokenizer()
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            pos_emb = clip(**inputs).last_hidden_state  # (batch, seq, dim)

        neg_inputs = tokenizer("", return_tensors="pt").to(device)
        with torch.no_grad():
            neg_emb = clip(**neg_inputs).last_hidden_state

        # Wrap conds into comfy expected format: list of (tensor, dict)
        # The dict can hold extra options if needed; minimal working format is {}
        positive = [(pos_emb, {})]
        negative = [(neg_emb, {})]

        # --- 2) Prepare noise / latent_image with time dimension ---
        # note: UNet expects latent channels = 4, and spatial dims are /8 of image dims
        h_lat = height // 8
        w_lat = width  // 8
        channels = 16  # WAN 2.2 latent space uses 16 channels, not 4
        latent_shape = (1, channels, frames, h_lat, w_lat)

        noise = torch.randn(latent_shape, device=device, dtype=dtype)
        latent_image = torch.zeros_like(noise)
        # Inject previous chunk latent tail (temporal carryover)
        if prev_latent_context is not None:
            overlap_frames = prev_latent_context.shape[2]
            latent_image[:, :, :overlap_frames] = prev_latent_context
            print(f"[WAN] Injected prev latent context ({overlap_frames} frames) into seed latent.")



        # --- 3) Call comfy sampler ---
        latents = comfy_sample.sample(
            model=unet,
            noise=noise,
            steps=steps,
            cfg=guidance,
            sampler_name="euler_a",#dpmpp_2m",
            scheduler="karras",
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=1.0,
            disable_pbar=False,
            seed=seed
        )

        # latents should be (1, 4, frames, h_lat, w_lat)
        print(f"[WAN] ✅ Latent generated. Shape: {tuple(latents.shape)}")
        return latents

    def decode_latent_to_mp4(self,
                         latent: torch.Tensor,
                         out_path: str,
                         fps: int = 16,
                         keep_frames: bool = True):
        """
        Decode WAN latent tensor → frames → MP4.
        Keeps PNG frames in a persistent folder (frames/<scene_name>/).
        Handles 16→4 projection if needed.
        """
        import imageio, tempfile, os, subprocess, shutil, numpy as np

        printt("[WAN] Decoding latent → frames → MP4 …")
        vae = self.pipe.get("vae", None)
        if vae is None:
            raise RuntimeError("VAE model missing in engine.pipe; please call load_models() first.")

        latent = latent.to(next(vae.parameters()).device)
        printt(f"[WAN] latent shape before decode: {tuple(latent.shape)}")

        T = latent.shape[2] if latent.dim() == 5 else 1
        if T == 0:
            raise RuntimeError("[WAN] ❌ Latent tensor has zero frames (T=0). Nothing to decode!")

        # --- Determine scene name and persistent output folder ---
        scene_name = os.path.splitext(os.path.basename(out_path))[0]
        frames_dir = os.path.join("frames", scene_name)
        os.makedirs(frames_dir, exist_ok=True)
        printt(f"[WAN] Saving decoded frames to: {frames_dir}")

        # --- Pre-create projection layer if needed ---
        needs_proj = latent.shape[1] == 16
        proj = None
        if needs_proj:
            printt("[WAN] ⚙️ Projecting latent channels 16 → 4 before VAE decode (applies to all frames)")
            proj = torch.nn.Conv2d(16, 4, kernel_size=1).to(latent.device, latent.dtype)

        saved_count = 0
        for i in range(T):
            print(f"[DEBUG] Starting  decoding 1 for {i}")
            with torch.no_grad():
                z = latent[:, :, i, :, :] if latent.dim() == 5 else latent
                if needs_proj:
                    z = proj(z)
                print(f"[DEBUG] Starting  decoding 2  for {i}")
                recon = vae.decode(z)
            print(f"[DEBUG] Starting  decoding 3  for {i}")
            if isinstance(recon, torch.Tensor):
                print(f"[DEBUG] Starting  decoding 4  for {i}")
                recon = recon.detach().cpu()
                if recon.min() < 0:  # normalize [-1,1] → [0,1]
                    recon = (recon.clamp(-1, 1) + 1) / 2
                frame = (recon[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if np.isnan(frame).any():
                    printt(f"[WARN] NaNs detected in frame {i}, skipping")
                    continue
                fname = os.path.join(frames_dir, f"frame_{i:04d}.png")
                print(f"[DEBUG] to be saved as {fname}")
                
                imageio.imwrite(fname, frame)
                print(f"[DEBUG] saved {fname}")
                saved_count += 1
            break
        if saved_count == 0:
            raise RuntimeError("[WAN] ❌ No frames saved — latent decode failed or returned NaNs")

        printt(f"[WAN] ✅ Saved {saved_count} frames, invoking ffmpeg @ {fps} fps …")

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
        ]
        subprocess.check_call(cmd)

        printt(f"[WAN] ✅ MP4 written: {out_path}")

        if not keep_frames:
            printt(f"[WAN] 🧹 Cleaning up frames in {frames_dir}")
            shutil.rmtree(frames_dir)

        return out_path


# ---- Audio helpers (Bark/XTTS/gTTS) ----
def make_tts_wav(text: str, out_wav: str, lang: str = 'en', tts_backend: str = 'bark', speaker: Optional[str] = None):
    out_wav = str(out_wav)
    if tts_backend == 'bark':
        try:
            from bark import generate_audio, preload_models
            preload_models()
            printt('Bark TTS: generating audio...')
            audio_arr = generate_audio(text)
            if SOUND_FILE_AVAILABLE:
                sf.write(out_wav, audio_arr, 24000)
            else:
                from scipy.io.wavfile import write as wavwrite
                wavwrite(out_wav, 24000, (audio_arr * 32767).astype('int16'))
            return out_wav
        except Exception as e:
            printt('Bark unavailable or failed:', e)
            tts_backend = 'xtts'
    if tts_backend == 'xtts':
        try:
            from TTS.api import TTS
            model_name = speaker or 'tts_models/multilingual/multi-dataset/your_model'
            tts = TTS(model_name)
            printt('XTTS: generating audio...')
            tts.tts_to_file(text=text, file_path=out_wav, speaker_wav=None, language=lang)
            return out_wav
        except Exception as e:
            printt('XTTS unavailable or failed:', e)
            tts_backend = 'gtts'
    if tts_backend == 'gtts':
        try:
            from gtts import gTTS
            printt('gTTS: generating audio...')
            tts = gTTS(text=text, lang=lang)
            tmp_mp3 = out_wav + '.tmp.mp3'
            tts.save(tmp_mp3)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_mp3, out_wav]
            subprocess.check_call(cmd)
            os.remove(tmp_mp3)
            return out_wav
        except Exception as e:
            printt('gTTS failed:', e)
            raise RuntimeError('No available TTS backend')

def mix_ambient(base_wav: str, ambient_wav: Optional[str], out_wav: str):
    if not ambient_wav:
        shutil.copy(base_wav, out_wav)
        return out_wav
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', base_wav,
        '-i', ambient_wav,
        '-filter_complex', "[0:a][1:a]amerge=inputs=2[a]", '-map', '[a]',
        '-ac', '2', out_wav
    ]
    subprocess.check_call(cmd)
    return out_wav

def mux_audio_video(video_mp4: str, audio_wav: str, out_path: str):
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', video_mp4,
        '-i', audio_wav,
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', out_path
    ]
    subprocess.check_call(cmd)
    return out_path

def adapt_to_aspect(input_mp4: str, target_w: int, target_h: int, out_mp4: Optional[str] = None) -> str:
    if out_mp4 is None:
        base = Path(input_mp4)
        out_mp4 = str(base.with_name(base.stem + f'_{target_h}p.mp4'))
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', input_mp4,
        '-vf', f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2",
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_mp4
    ]
    subprocess.check_call(cmd)
    return out_mp4

# ---- latent cache utilities and refine ----
def save_latent(latent: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent.cpu(), path)
    printt('Saved latent:', path)

def load_latent(path: Path) -> torch.Tensor:
    return torch.load(path)

def refine_latent(latent: torch.Tensor,
                  feedback_prompt: str = "",
                  strength: float = 0.35,
                  camera_hint: Optional[str] = None,
                  lighting_hint: Optional[str] = None) -> torch.Tensor:
    """
    Safe latent refinement (works for 3D, 4D, or 5D tensors).
    Adds temporal smoothing and mean normalization.
    """
    latent = latent.clone()
    dtype = latent.dtype
    if dtype == torch.float16:
        latent = latent.float()

    # --- Normalize dimensions ---
    # Expected: (B, C, T, H, W)
    if latent.dim() == 5:
        pass  # correct shape
    elif latent.dim() == 4:
        latent = latent.unsqueeze(2)  # (B, C, 1, H, W)
    elif latent.dim() == 3:
        latent = latent.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
    else:
        raise ValueError(f"[WAN] Unexpected latent shape {latent.shape}")

    p = (feedback_prompt or "").lower()
    kernel = 5 if ("smooth" in p or "gentle" in p) else 3
    pad = kernel // 2

    # --- Temporal smoothing (3-frame average) ---
    if latent.shape[2] > 1:
        smoothed = F.avg_pool3d(latent, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        latent = (1 - strength * 0.6) * latent + (strength * 0.6) * smoothed

    # --- Mean balance across frames ---
    mean_per_frame = latent.mean(dim=[1, 2, 3, 4], keepdim=True)
    global_mean = mean_per_frame.mean()
    latent = latent + (global_mean - mean_per_frame) * 0.3

    # --- Optional: Camera & zoom hints ---
    if camera_hint:
        if "pan left" in camera_hint:
            latent = torch.roll(latent, shifts=-2, dims=-1)
        elif "pan right" in camera_hint:
            latent = torch.roll(latent, shifts=2, dims=-1)

    if "zoom in" in p:
        latent = latent * 1.02
    elif "zoom out" in p:
        latent = latent * 0.98

    latent = latent.clamp(-10.0, 10.0)
    if dtype == torch.float16:
        latent = latent.half()

    return latent


def generate_chunked_latents(engine, prompt, total_frames=64, chunk_size=16, overlap=4,
                            width=1280, height=720, steps=12, guidance=6.5, seed=None):
    """
    engine.sample_latent must accept frames argument (we force it to generate chunk_size frames).
    This function generates chunks with overlap and stitches them by crossfade.
    Returns: full_latents of shape (1, C, total_frames, H, W)
    """
    # === replacement for generate_chunked_latents ===
def generate_chunked_latents(engine, prompt, total_frames=64, chunk_size=16, overlap=4,
                            width=1280, height=720, steps=12, guidance=6.5, seed=None):
    """
    Sequential chunked generation with latent overlap carryover.
    Each new chunk uses the tail latents of the previous chunk to maintain temporal coherence.
    """
    assert overlap < chunk_size
    stride = chunk_size - overlap
    chunks = []
    base_seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
    prev_context = None

    for start in range(0, total_frames, stride):
        remaining = total_frames - start
        cur_size = min(chunk_size, remaining)
        seed_i = base_seed + (start // stride)

        printt(f"[WAN] Generating chunk {start // stride + 1}: frames={cur_size}, seed={seed_i}")
        chunk_latent = engine.sample_latent(
            prompt=prompt,
            frames=cur_size,
            width=width,
            height=height,
            guidance=guidance,
            steps=steps,
            seed=seed_i,
            prev_latent_context=prev_context
        )

        # Prepare context for next chunk (use last `overlap` frames)
        prev_context = chunk_latent[:, :, -overlap:].detach().clone()
        chunks.append(chunk_latent)

    # Concatenate all chunks along the temporal dimension
    stitched = torch.cat(
        [chunks[0]] + [c[:, :, overlap:] for c in chunks[1:]],
        dim=2
    )

    printt(f"[WAN] ✅ Full latent stitched: {tuple(stitched.shape)}")
    return stitched

# ---- render_scene / flows ----

def render_scene(engine: WANEngine, prompt: str, out_base: str, frames: int, width: int, height: int, guidance: float, motion: float, noise_ratio: float, steps: int, seed: Optional[int], ip_emb: Optional[torch.Tensor], audio_spec: Optional[Dict[str,Any]], persona_name: Optional[str], tts_backend: str, fps: int = DEFAULT_CONFIG['default_fps']) -> Dict[str, Any]:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    printt(f"[WAN] Rendering scene -> out_base={out_base}, frames={frames}, {width}x{height}, steps={steps}, seed={seed}")

    # --- generate latents in chunks (use correct kwarg name total_frames) ---
    latent = generate_chunked_latents(
        engine=engine,
        prompt=prompt,
        total_frames=frames,   # <--- was 'frames=' previously (bug)
        chunk_size=16,
        overlap=4,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed
    )

    latent_path = CACHE_DIR / f"{out_base.name}.latent.pt"
    save_latent(latent, latent_path)

    # --- refine latent (in-place safe copy) ---
    printt("[WAN] Refining latent...")
    refined = refine_latent(latent, feedback_prompt="", strength=0.28)
    refined_path = CACHE_DIR / f"{out_base.name}.refined.latent.pt"
    save_latent(refined, refined_path)
    printt("[WAN] Saved refined latent:", refined_path)

    # --- decode refined latent to frames using loaded VAE (per-frame decode to save memory) ---
    vae = engine.pipe.get("vae", None)
    if vae is None:
        raise RuntimeError("No VAE model loaded in engine.pipe — ensure load_models() loaded wan_2.1_vae.safetensors")

    full_latents = refined
    total_frames = int(full_latents.shape[2])
    decoded_frames = []

    printt(f"[WAN] Decoding {total_frames} frames (per-frame) with VAE...")

    for i in range(total_frames):
        latent_frame = full_latents[:, :, i:i+1, :, :]  # (1, C, 1, H, W)
        lf = latent_frame.squeeze(2)                    # -> (1, C, H, W)
        with torch.no_grad():
            # decode should return PIL-like or tensor depending on your VAE; adapt if required
            img = vae.decode(lf)   # expected to return tensor shape (1, 3, H_img, W_img) in [-1,1] or [0,1]
        # normalize/convert to numpy image (attempt common formats)
        if isinstance(img, torch.Tensor):
            img_cpu = img.detach().cpu()
            # If VAE returns in [-1,1], convert to [0,255]
            if img_cpu.min() < 0.0:
                frame_np = ((img_cpu.clamp(-1,1) + 1.0) / 2.0).numpy()
            else:
                frame_np = img_cpu.numpy()
            # frame_np shape (1, C, H, W)
            frame_rgb = (np.clip(frame_np[0].transpose(1, 2, 0) * 255.0, 0, 255)).astype(np.uint8)
            decoded_frames.append(frame_rgb)
        else:
            # assume PIL.Image returned
            decoded_frames.append(np.array(img))

    printt(f"[WAN] ✅ Decoded {len(decoded_frames)} frames")

    # --- save frames to temporary dir for ffmpeg ---
    import imageio, numpy as np
    tmpd = tempfile.mkdtemp(prefix=f"wan_decode_{out_base.name}_")
    for i, frame in enumerate(decoded_frames):
        path = os.path.join(tmpd, f"frame_{i:04d}.png")
        imageio.imwrite(path, frame)
    printt(f"[WAN] Saved frames to {tmpd}")

    # --- assemble mp4 (no audio) ---
    mp4_noaudio = str(out_base.with_suffix('.noaudio.mp4'))
    video_fps = fps if fps and fps > 0 else max(1, int(total_frames / 4))
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(video_fps),
        '-i', os.path.join(tmpd, 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', mp4_noaudio
    ]
    printt('Running ffmpeg to produce mp4 (no audio) for', out_base)
    subprocess.check_call(cmd)
    printt('[WAN] Produced mp4 (no audio):', mp4_noaudio)

    # cleanup frames temp dir
    shutil.rmtree(tmpd)

    # --- handle audio (dialogue / ambient) and mux ---
    final_mp4 = str(out_base.with_suffix('.final.mp4'))

    if audio_spec and ('dialogue' in audio_spec or 'audio_summary' in audio_spec):
        dialogue = audio_spec.get('dialogue')
        audio_summary = audio_spec.get('audio_summary')
        dialogue_text = None
        tts_lang = 'en'
        if isinstance(dialogue, dict):
            dialogue_text = dialogue.get('text')
            tts_lang = dialogue.get('lang', 'en')
        elif isinstance(dialogue, str):
            dialogue_text = dialogue

        dialogue_wav = None
        if dialogue_text:
            dialogue_wav = str(out_base.with_suffix('.dialogue.wav'))
            printt("[WAN] Generating TTS for dialogue...")
            make_tts_wav(dialogue_text, dialogue_wav, lang=tts_lang, tts_backend=tts_backend)

        ambient_wav = None
        if audio_summary and isinstance(audio_summary, str):
            candidate = ASSETS_DIR / (audio_summary + '.wav')
            if candidate.exists():
                ambient_wav = str(candidate)

        mixed_wav = str(out_base.with_suffix('.mixed.wav'))

        if dialogue_wav:
            printt("[WAN] Mixing dialogue with ambient (if present)...")
            mix_ambient(dialogue_wav, ambient_wav, mixed_wav)
            printt("[WAN] Muxing audio/video...")
            mux_audio_video(mp4_noaudio, mixed_wav, final_mp4)
        else:
            if ambient_wav:
                mux_audio_video(mp4_noaudio, ambient_wav, final_mp4)
            else:
                shutil.copy(mp4_noaudio, final_mp4)
    else:
        # no audio requested → copy noaudio to final
        shutil.copy(mp4_noaudio, final_mp4)

    printt("[WAN] Final video at:", final_mp4)

    return {"latent_path": str(latent_path), "refined_path": str(refined_path), "final_mp4": final_mp4}


def generate_clip(prompt: str, out_name: str, seed: Optional[int] = None, frames: Optional[int] = None, resolution: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None, precision: str = 'fp16', guidance_scale: float = 7.5, motion_strength: float = 1.0, noise_ratio: float = 0.5, steps: int = 20, gpu_profile: Optional[str] = None, use_control_adapter: bool = False, ip_adapter_emb: Optional[torch.Tensor] = None, persona: Optional[str] = None, tts_backend: str = 'bark') -> Dict[str, Any]:
    frames = frames or DEFAULT_CONFIG['default_frames']
    width, height = get_resolution(resolution, width, height)
    out_base = Path(out_name)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    profile = detect_gpu_profile(gpu_profile)
    printt('Hardware profile:', profile)
    engine = WANEngine(models_dir=MODELS_DIR, device=profile['device'], precision=precision)
    engine.load_models(high=DEFAULT_MODELS['unet_high'], low=DEFAULT_MODELS['unet_low'], vae=DEFAULT_MODELS['vae'], text_encoder=DEFAULT_MODELS['text_encoder'])
    audio_spec = {'dialogue': {'text': prompt, 'lang': 'en'}}
    res = render_scene(engine, prompt=prompt, out_base=str(out_base), frames=frames, width=width, height=height, guidance=guidance_scale, motion=motion_strength, noise_ratio=noise_ratio, steps=steps, seed=seed, ip_emb=ip_adapter_emb, audio_spec=audio_spec, persona_name=persona, tts_backend=tts_backend, fps=DEFAULT_CONFIG['default_fps'])
    adapted = adapt_to_aspect(res['final_mp4'], width, height, out_mp4=str(out_base.with_suffix('.final.adapted.mp4')))
    if profile['device'] == 'cuda':
        engine.unload_models()
    return {**res, 'adapted_final_mp4': adapted}

def refine_clip(latent_path: str, feedback: str, out_name: str, strength: float = 0.35, precision: str = 'fp16', tts_backend: str = 'bark') -> Dict[str, Any]:
    latent = load_latent(Path(latent_path))
    refined = refine_latent(latent, feedback_prompt=feedback, strength=strength)
    refined_path = Path(CACHE_DIR) / f"{out_name}.refined.latent.pt"
    save_latent(refined, refined_path)
    engine = WANEngine(models_dir=MODELS_DIR, device=detect_gpu_profile().get('device','cpu'), precision=precision)
    mp4_out = Path(f"outputs/{out_name}.refined.mp4")
    mp4_out.parent.mkdir(parents=True, exist_ok=True)
    frames_pil = engine.decode_latent_to_frames(refined)
    tmpd = tempfile.mkdtemp(prefix=f"wan_refine_{out_name}_")
    for i, pil in enumerate(frames_pil):
        pil.save(os.path.join(tmpd, f"frame_{i:04d}.png"))
    cmd = ['ffmpeg','-y','-loglevel','error','-framerate',str(DEFAULT_CONFIG['default_fps']),'-i',os.path.join(tmpd,'frame_%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(mp4_out)]
    subprocess.check_call(cmd)
    shutil.rmtree(tmpd)
    return {"refined_latent": str(refined_path), "refined_mp4": str(mp4_out)}

def render_script(engine: WANEngine, script: Dict[str, Any], output_dir: str = 'outputs', persona_name: Optional[str] = None, tts_backend: str = 'bark', resolution: Optional[str] = None) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    ip = IPAdapterCLIP(device='cpu')
    persona_emb = None
    if persona_name:
        persona_emb = ip.load_persona(persona_name)
    for scene in script.get('scenes', []):
        sid = scene.get('id', f"scene_{random.randint(0,9999)}")
        summary = scene.get('summary', '')
        camera = scene.get('camera', '')
        lighting = scene.get('lighting', '')
        duration = float(scene.get('duration', 6.0))
        frames = int(DEFAULT_CONFIG['default_fps'] * duration)
        prompt = f"{summary}. Camera: {camera}. Lighting: {lighting}"
        ip_emb = persona_emb
        char_ref = scene.get('character')
        if char_ref and not ip_emb:
            ip_emb = ip.load_persona(char_ref)
        out_base = Path(output_dir) / sid
        res = render_scene(engine, prompt=prompt, out_base=str(out_base), frames=frames, width=get_resolution(resolution, None, None)[0], height=get_resolution(resolution, None, None)[1], guidance=7.5, motion=1.0, noise_ratio=0.5, steps=20, seed=None, ip_emb=ip_emb, audio_spec={'dialogue': scene.get('dialogue'), 'audio_summary': scene.get('audio_summary')}, persona_name=persona_name, tts_backend=tts_backend)
        results.append({"scene_id": sid, **res})
    if results:
        concat_txt = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        for r in results:
            concat_txt.write(f"file '{os.path.abspath(r['final_mp4'])}'\n")
        concat_txt.close()
        final_comp = Path(output_dir) / 'final_compilation.mp4'
        cmd = ['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',concat_txt.name,'-c','copy',str(final_comp)]
        printt('Concatenating scenes into final compilation...')
        subprocess.check_call(cmd)
        os.unlink(concat_txt.name)
        return {"scenes": results, "final_compilation": str(final_comp)}
    return {"scenes": results}

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['init_persona','generate','refine','render_script'], required=True)
    p.add_argument('--prompt', type=str, default='')
    p.add_argument('--out', type=str, required=False, default='out')
    p.add_argument('--latent', type=str, default='')
    p.add_argument('--feedback', type=str, default='')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--frames', type=int, default=None)
    p.add_argument('--resolution', choices=list(RES_PRESETS.keys()), default=None)
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--precision', choices=['fp32','fp16'], default='fp16')
    p.add_argument('--guidance', type=float, default=7.5)
    p.add_argument('--motion', type=float, default=1.0)
    p.add_argument('--noise_ratio', type=float, default=0.5)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--gpu', type=str, default=None)
    p.add_argument('--persona', type=str, default=None)
    p.add_argument('--ref_dir', type=str, default=None, help='When init_persona: directory of reference images')
    p.add_argument('--script_json', type=str, default='')
    p.add_argument('--tts', choices=['bark','xtts','gtts'], default='bark')
    return p.parse_args()

def main():
    args = parse_args()
    profile = detect_gpu_profile(args.gpu)
    device = profile['device']
    printt('Using device:', device)
    if args.mode == 'init_persona':
        if not args.persona or not args.ref_dir:
            raise RuntimeError('--persona and --ref_dir required for init_persona')
        ip = IPAdapterCLIP(device=device)
        img_paths = [str(p) for p in Path(args.ref_dir).iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg']]
        if not img_paths:
            raise RuntimeError('No images found in ref_dir')
        emb = ip.batch_compute(img_paths)
        ip.save_persona(args.persona, emb.mean(dim=0, keepdim=True))
        printt('Persona initialized and saved:', args.persona)
        return
    if args.mode == 'generate':
        
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
            use_control_adapter=False,
            ip_adapter_emb=None,
            persona=args.persona,
            tts_backend=args.tts,
        )
        printt('Generated:', json.dumps(res, indent=2))
        return
    if args.mode == 'refine':
        if not args.latent:
            raise RuntimeError('Please provide --latent for refine mode')
        res = refine_clip(args.latent, args.feedback or args.prompt, args.out, strength=0.35, precision=args.precision, tts_backend=args.tts)
        printt('Refined:', json.dumps(res, indent=2))
        return
    if args.mode == 'render_script':
        if not args.script_json:
            raise RuntimeError('--script_json is required for render_script mode')
        with open(args.script_json, 'r') as f:
            script = json.load(f)
        engine = WANEngine(models_dir=MODELS_DIR, device=device, precision=args.precision)
        engine.load_models(high=DEFAULT_MODELS['unet_high'], low=DEFAULT_MODELS['unet_low'], vae=DEFAULT_MODELS['vae'], text_encoder=DEFAULT_MODELS['text_encoder'])
        res = render_script(engine, script, output_dir=args.out, persona_name=args.persona, tts_backend=args.tts, resolution=args.resolution)
        printt('Render script completed:', json.dumps(res, indent=2))
        if device == 'cuda':
            engine.unload_models()
        return

if __name__ == '__main__':
    main()
