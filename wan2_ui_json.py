#!/usr/bin/env python3
import streamlit as st, json, tempfile, os
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips
from wan2_workflow_v2 import generate_clip, refine_clip

st.set_page_config(page_title="WAN2.2 Scene Generator + Refine", layout="wide")
st.title("🎬 WAN2.2 Scene-to-Video + Feedback Refinement (A40 Ready)")

# Session tracking for refinement
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "final_compiled" not in st.session_state:
    st.session_state.final_compiled = None

# -----------------------------
# Scene-based JSON editor
# -----------------------------
default_json = {
    "scenes": [
        {
            "id": "scene1",
            "summary": "Dog and cat walking through a sunset field",
            "audio_summary": "soft wind and gentle tone",
            "dialogue": "Dog: What a beautiful day! Cat: Yes, perfect for a walk.",
            "camera": "slow pan right, warm light",
            "duration": 6.0
        },
        {
            "id": "scene2",
            "summary": "Cat jumps onto rock, dog laughs",
            "audio_summary": "light laughter, playful tone",
            "dialogue": "Cat: Look, I can climb higher! Dog: Don’t fall!",
            "camera": "tracking shot",
            "duration": 5.0
        }
    ]
}

tabs = st.tabs(["🧠 Generate Scenes", "🎨 Refine Feedback"])

# -----------------------------
# Tab 1: Generate
# -----------------------------
with tabs[0]:
    st.subheader("📜 Scene JSON Script")
    script_json = st.text_area(
        "Edit or paste your scene JSON:",
        value=json.dumps(default_json, indent=2),
        height=400
    )

    col1, col2 = st.columns(2)
    with col1:
        resolution = st.selectbox("Resolution", ["480p", "720p", "1080p"], index=1)
        guidance = st.slider("Guidance", 1.0, 10.0, 7.5)
        motion = st.slider("Motion", 0.5, 2.0, 1.0)
        noise = st.slider("Noise Ratio", 0.1, 1.0, 0.5)
    with col2:
        steps = st.slider("Steps", 10, 40, 20)
        gpu = st.selectbox("GPU", ["auto", "a40", "cpu"], index=1)
        precision = st.selectbox("Precision", ["fp16", "fp32"], index=0)
        run_btn = st.button("🚀 Generate Full Video")

    if run_btn:
        data = json.loads(script_json)
        clips = []
        temp_dir = Path(tempfile.mkdtemp(prefix="wan2_scenes_"))
        results = []
        for scene in data["scenes"]:
            st.info(f"🎬 Generating {scene['id']}: {scene['summary']}")
            result = generate_clip(
                prompt=f"{scene['summary']}, {scene['camera']}",
                out_name=f"{scene['id']}",
                resolution=resolution,
                guidance_scale=guidance,
                motion_strength=motion,
                noise_ratio=noise,
                steps=steps,
                gpu_profile=None if gpu == "auto" else gpu,
                precision=precision,
            )
            st.video(result["final_mp4"])
            
            results.append(result)
            clips.append(VideoFileClip(result["final_mp4"]))

        st.session_state.last_results = results

        if len(clips) > 1:
            final_path = temp_dir / "final_compiled.mp4"
            st.info("🧩 Combining scenes...")
            final = concatenate_videoclips(clips)
            final.write_videofile(str(final_path), codec="libx264", audio_codec="aac", fps=16)
            st.session_state.final_compiled = str(final_path)
            st.success("✅ Final video complete!")
            st.video(str(final_path))

# -----------------------------
# Tab 2: Refine
# -----------------------------
with tabs[1]:
    st.subheader("🎨 Feedback-Based Refinement")

    if not st.session_state.last_results:
        st.warning("⚠️ No generated scenes found! Please generate first.")
    else:
        refine_options = [res["final_mp4"] for res in st.session_state.last_results]
        choice = st.selectbox("Select Scene to Refine", refine_options)
        feedback = st.text_area(
            "Enter feedback prompt (e.g. 'Make lighting warmer and smoother camera')",
            height=120
        )
        refine_strength = st.slider("Refinement Strength", 0.1, 1.0, 0.35)
        refine_btn = st.button("🎨 Apply Refinement")

        if refine_btn:
            match = next((r for r in st.session_state.last_results if r["final_mp4"] == choice), None)
            if not match:
                st.error("❌ Could not find latent for selected video.")
            else:
                latent_path = match["latent_path"]
                st.info(f"🎨 Refining {choice} using latent: {latent_path}")
                refined = refine_clip(latent_path, feedback, out_name="refined_ui", strength=refine_strength)
                st.success("✅ Refinement complete!")
                st.video(refined["final_mp4"])
