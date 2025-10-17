# wan2_ui_v3.py - Streamlit UI for wan2_workflow_v3
import streamlit as st
import json, os, subprocess, tempfile, shutil
from pathlib import Path

st.set_page_config(layout="wide", page_title="WAN2 Workflow V3 UI")

st.title("WAN2 Workflow V3 — JSON Scenes & Refine")

col1, col2 = st.columns([2,1])

with col1:
    tab = st.tabs(["Generate Single", "Render Script", "Refine", "Persona Init"])[0]

with col2:
    st.info("A40-ready UI: make sure you ran setup.sh and activated the venv.\nStreamlit runs inside that venv.")

# Generate Single
with st.expander("Generate Single Scene", expanded=True):
    prompt = st.text_area("Prompt / Scene summary", value="Garuda flies over a golden temple at sunset, majestic and powerful", height=120)
    out_name = st.text_input("Output base name", value="scene_garuda_01")
    resolution = st.selectbox("Resolution", options=["720p","1080p","480p"], index=1)
    tts = st.selectbox("TTS Backend", options=["bark","xtts","gtts"], index=0)
    st.write("GPU override (optional): 'a40', '4090', 'cpu' etc.")
    gpu_override = st.text_input("GPU override", value="")
    generate_btn = st.button("🚀 Generate Scene (runs wan2_workflow_v3.py)")

if generate_btn:
    cmd = [
        "python", "wan2_workflow_v3.py",
        "--mode", "generate",
        "--prompt", prompt,
        "--out", out_name,
        "--resolution", resolution,
        "--tts", tts
    ]
    if gpu_override:
        cmd += ["--gpu", gpu_override]
    with st.spinner("Running generation (this may take a while on first run)..."):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        st.text(proc.stdout)
        if proc.returncode != 0:
            st.error("Generation failed. See logs below.")
            st.code(proc.stderr)
        else:
            st.success("Generation finished")
            mp4 = Path(f"{out_name}.final.adapted.mp4")
            if mp4.exists():
                st.video(str(mp4))
            else:
                mp42 = Path(f"{out_name}.final.mp4")
                if mp42.exists():
                    st.video(str(mp42))
                else:
                    st.warning("No mp4 found in expected outputs.")

# Render Script
with st.expander("Render Script (JSON)", expanded=False):
    script_text = st.text_area("Scene JSON (array of scenes)", value=json.dumps({
        "scenes":[
            {"id":"scene1","summary":"Garuda descends on a temple hill","dialogue":{"lang":"en","text":"Behold the mighty Garuda!"},"audio_summary":"wind_bells","camera":"slow pan right","lighting":"sunset orange","duration":8.0},
            {"id":"scene2","summary":"Garuda meets Vishnu","dialogue":{"lang":"en","text":"O Lord, I bring news."},"audio_summary":"temple_bells","camera":"medium tracking","lighting":"soft golden","duration":6.0}
        ]
    }, indent=2), height=240)
    out_dir = st.text_input("Output directory", value="outputs")
    persona = st.text_input("Persona name (optional)", value="")
    tts = st.selectbox("TTS Backend (script)", options=["bark","xtts","gtts"], index=0, key="tts_script")
    render_btn = st.button("🎬 Render Script")

if render_btn:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(script_text.encode("utf-8"))
    tmp.close()
    cmd = [
        "python", "wan2_workflow_v3.py",
        "--mode", "render_script",
        "--script_json", tmp.name,
        "--out", out_dir,
        "--tts", tts
    ]
    if persona:
        cmd += ["--persona", persona]
    with st.spinner("Rendering scenes..."):
        proc = subprocess.run(cmd, text=True, capture_output=True)
        st.text(proc.stdout)
        if proc.returncode != 0:
            st.error("Render failed")
            st.code(proc.stderr)
        else:
            st.success("Render finished")
            final = Path(out_dir) / "final_compilation.mp4"
            if final.exists():
                st.video(str(final))
            else:
                st.warning("No final compilation found - check outputs folder")

# Refine
with st.expander("Refine Cached Latent", expanded=False):
    latent_path = st.text_input("Latent (.pt) path", value="")
    feedback = st.text_area("Refinement instructions (e.g. 'make lighting warmer, smooth wing motion')", value="", height=120)
    out_name_ref = st.text_input("Refined output base name", value="refined_scene")
    tts = st.selectbox("TTS Backend (refine)", options=["bark","xtts","gtts"], index=0, key="tts_refine")
    refine_btn = st.button("🛠️ Apply Refinement")

if refine_btn:
    if not latent_path:
        st.warning("Provide latent path")
    else:
        cmd = ["python", "wan2_workflow_v3.py", "--mode", "refine", "--latent", latent_path, "--feedback", feedback, "--out", out_name_ref, "--tts", tts]
        with st.spinner("Running refine..."):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            st.text(proc.stdout)
            if proc.returncode != 0:
                st.error("Refine failed")
                st.code(proc.stderr)
            else:
                st.success("Refine finished")
                mp4 = Path(f"outputs/{out_name_ref}.refined.mp4")
                if mp4.exists():
                    st.video(str(mp4))
                else:
                    st.warning("Refined mp4 not found. Check outputs/")

# Persona init
with st.expander("Persona Init (compute and cache CLIP embeddings)", expanded=False):
    persona_name = st.text_input("Persona name", value="garuda")
    ref_dir = st.text_input("Reference images directory (on server)", value="ref/garuda")
    init_btn = st.button("🧠 Init Persona")

if init_btn:
    if not persona_name or not ref_dir:
        st.warning("Provide persona name and ref_dir")
    else:
        cmd = ["python", "wan2_workflow_v3.py", "--mode", "init_persona", "--persona", persona_name, "--ref_dir", ref_dir]
        with st.spinner("Computing persona embeddings..."):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            st.text(proc.stdout)
            if proc.returncode != 0:
                st.error("Persona init failed")
                st.code(proc.stderr)
            else:
                st.success("Persona saved to persona_cache/")
