#!/usr/bin/env python3
"""
wan2_workflow.py
Unified workflow for WAN2.2 + Bark + scene-based video generation.
"""

import os, subprocess, json, numpy as np, torch
from pathlib import Path
from dataclasses import dataclass, asdict
from wan2_workflow_v2 import generate_clip, refine_clip
from bark import generate_audio as bark_generate
from bark.generation import SAMPLE_RATE
import soundfile as sf, numpy as np
# ------------------ CONFIGURATION ------------------
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "fp16": True,
    "memory_saver": True,
    "wan_model_high": "./models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "wan_model_low": "./models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    "vae_model": "./models/wan_2.1_vae.safetensors",
    "default_width": 512,
    "default_height": 512,
    "default_frames": 64,
    "fps": 16,
    "output_dir": "./output",
    "audio_sample_rate": 24000,
    "video_codec": "libx264",
    "gpu_name": "A40",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ------------------ DATA MODEL ------------------
@dataclass
class SceneData:
    id: str
    summary: str
    audio_summary: str
    dialogue: str
    camera: str
    duration: float

# ------------------ WAN 2.2 VIDEO ------------------
def generate_video(scene: SceneData, out_dir: Path):
    """
    Placeholder for WAN 2.2 generation.
    Replace with your WAN2 API call.
    """
    video_out = out_dir / f"{scene.id}_video.mp4"
    print(f"[WAN] Generating video for: {scene.summary} ({scene.duration}s)")
    
    result = generate_clip(prompt="Sunrise over lake", out_name="scene1")
    # (Placeholder video)
    '''subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=gray:s={CONFIG['default_width']}x{CONFIG['default_height']}:d={scene.duration}",
        "-vf", f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='{scene.summary}':x=50:y=50:fontsize=20:fontcolor=white",
        "-c:v", CONFIG["video_codec"],
        "-pix_fmt", "yuv420p", str(video_out)
    ])
    '''
    return result

# ------------------ BARK AUDIO ------------------
def generate_audio(scene: SceneData, out_dir: Path):
    

    out_wav = out_dir / f"{scene.id}_audio.wav"
    text = f"{scene.audio_summary}. {scene.dialogue}"
    print(f"[BARK] Generating audio for scene {scene.id}")

    audio_array = bark_generate(text)
    audio = np.array(audio_array, dtype=np.float32)
    actual_len = len(audio) / SAMPLE_RATE
    target_len = int(scene.duration * SAMPLE_RATE)

    if actual_len > scene.duration:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    sf.write(out_wav, audio, SAMPLE_RATE)
    return out_wav

# ------------------ UTILITIES ------------------
def match_video_to_audio(video_path, audio_path, out_path):
    data, sr = sf.read(audio_path)
    aud_len = len(data) / sr
    vid_len = float(subprocess.check_output([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",video_path
    ]).decode())
    ratio = aud_len / max(vid_len, 1e-3)
    subprocess.run([
        "ffmpeg","-y","-i",video_path,
        "-filter:v",f"setpts={1/ratio}*PTS",
        "-an",str(out_path)
    ])
    return out_path

def mux_audio_video(video, audio, out):
    subprocess.run([
        "ffmpeg","-y","-i",video,"-i",audio,
        "-c:v","copy","-c:a","aac","-shortest",str(out)
    ])
    return out

def stitch_videos(video_files, output_file):
    txt = Path("concat_list.txt")
    with open(txt,"w") as f:
        for v in video_files:
            f.write(f"file '{v}'\n")
    subprocess.run([
        "ffmpeg","-y","-f","concat","-safe","0","-i",str(txt),
        "-c","copy",str(output_file)
    ])
    txt.unlink()
    return output_file

# ------------------ MAIN PIPELINE ------------------
def process_scene(scene: SceneData):
    out_dir = Path(CONFIG["output_dir"])
    video = generate_video(scene, out_dir)
    audio = generate_audio(scene, out_dir)
    timed = out_dir / f"{scene.id}_timed.mp4"
    final = out_dir / f"{scene.id}_final.mp4"

    match_video_to_audio(video, audio, timed)
    mux_audio_video(timed, audio, final)
    return str(final)

def process_script(script_json):
    scenes = [SceneData(**sc) for sc in script_json["scenes"]]
    results = []
    for sc in scenes:
        results.append(process_scene(sc))
    stitched = Path(CONFIG["output_dir"]) / "full_movie.mp4"
    stitch_videos(results, stitched)
    print(f"✅ Final stitched movie: {stitched}")
    return stitched



def dispatch_to_wanr2(task):
    if task["mode"] == "refine":
        return refine_clip(
            latent_path=task["latent"],
            feedback_prompt=task["feedback"],
            out_name=task["out"],
            strength=0.35,
            precision="fp16"
        )