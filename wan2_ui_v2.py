#!/usr/bin/env python3
"""
wan2_ui_v2.py

Interactive CLI + video preview UI for WAN2.2 workflow.
Handles:
  - Full parameter configuration for generation
  - Automatic latent tracking
  - Integrated refine loop
  - Inline video playback

Dependencies:
  pip install rich moviepy
"""

import json, os, subprocess, sys
from pathlib import Path
from typing import Any, Dict

from rich.prompt import Prompt, Confirm
from rich.console import Console
from rich.table import Table
from moviepy.editor import VideoFileClip

# backend imports
from wan2_workflow_v2 import generate_clip, refine_clip, CONFIG, printt

console = Console()
SESSION_FILE = Path("session.json")

# -----------------------
# Helpers
# -----------------------
def save_session(data: Dict[str, Any]):
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_session() -> Dict[str, Any]:
    if not SESSION_FILE.exists():
        return {}
    with open(SESSION_FILE) as f:
        return json.load(f)

def play_video(path: str):
    """Plays video inline (moviepy preview) or fallback to system player"""
    try:
        clip = VideoFileClip(path)
        clip.preview()   # plays video in a window
    except Exception as e:
        console.print(f"[red]Inline preview failed ({e}), opening externally...[/red]")
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

# -----------------------
# UI Input Functions
# -----------------------
def configure_parameters() -> Dict[str, Any]:
    """Ask user for generation parameters"""
    console.rule("[bold cyan]WAN2.2 Generate Configuration[/bold cyan]")
    prompt = Prompt.ask("🧠 Enter your text prompt")
    out_name = Prompt.ask("💾 Output base name", default="clip1")
    resolution = Prompt.ask("📺 Resolution (480p/720p/1080p/4k)", default="720p")
    guidance = float(Prompt.ask("🎚️ Guidance scale", default="7.5"))
    motion = float(Prompt.ask("🎞️ Motion strength", default="1.0"))
    noise_ratio = float(Prompt.ask("🌫️ Noise ratio", default="0.5"))
    steps = int(Prompt.ask("🪜 Diffusion steps", default="20"))
    gpu = Prompt.ask("⚙️ GPU profile (a40/4090/cpu)", default="auto")
    precision = Prompt.ask("📏 Precision (fp16/fp32/fp8)", default="fp16")
    use_control = Confirm.ask("🎮 Use control adapter?", default=True)

    return {
        "mode": "generate",
        "prompt": prompt,
        "out": out_name,
        "resolution": resolution,
        "guidance": guidance,
        "motion": motion,
        "noise_ratio": noise_ratio,
        "steps": steps,
        "gpu": None if gpu == "auto" else gpu,
        "precision": precision,
        "use_control_adapter": use_control,
    }

def configure_refine(session: Dict[str, Any]) -> Dict[str, Any]:
    """Ask user for refinement feedback and re-run"""
    console.rule("[bold magenta]WAN2.2 Refinement[/bold magenta]")
    if "last_latent" not in session:
        console.print("[red]No previous generation found! Please run generate first.[/red]")
        sys.exit(1)

    feedback = Prompt.ask("💬 Feedback prompt (e.g. 'make lighting warmer')")
    out_name = Prompt.ask("💾 Output name for refined clip", default=session.get("last_out", "refined_clip"))

    return {
        "mode": "refine",
        "latent": session["last_latent"],
        "feedback": feedback,
        "out": out_name,
        "precision": session.get("precision", "fp16"),
    }

# -----------------------
# Main UI Logic
# -----------------------
def main_menu():
    console.rule("[bold green]WAN2_UI v2[/bold green]")
    table = Table(title="Options")
    table.add_column("Mode", style="cyan")
    table.add_column("Description", style="yellow")
    table.add_row("1", "Generate a new video")
    table.add_row("2", "Refine last generated video")
    table.add_row("3", "Exit")
    console.print(table)
    choice = Prompt.ask("Select option", choices=["1", "2", "3"])
    return choice

def main():
    session = load_session()

    while True:
        choice = main_menu()

        if choice == "1":
            params = configure_parameters()
            result = generate_clip(**params)
            console.print("[bold green]✅ Generation complete![/bold green]")
            console.print_json(json.dumps(result, indent=2))

            session["last_latent"] = result["latent_path"]
            session["last_out"] = params["out"]
            session["precision"] = params["precision"]
            save_session(session)

            play = Confirm.ask("▶️ Play generated video?", default=True)
            if play:
                play_video(result["final_mp4"])

        elif choice == "2":
            refine_task = configure_refine(session)
            result = refine_clip(**refine_task)
            console.print("[bold magenta]🎨 Refinement complete![/bold magenta]")
            console.print_json(json.dumps(result, indent=2))

            session["last_latent"] = result["refined_latent"]
            session["last_out"] = refine_task["out"]
            save_session(session)

            play = Confirm.ask("▶️ Play refined video?", default=True)
            if play:
                play_video(result["final_mp4"])

        else:
            console.print("[bold yellow]Exiting WAN2_UI v2.[/bold yellow]")
            break


if __name__ == "__main__":
    main()
