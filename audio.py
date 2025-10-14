from bark import preload_models, generate_audio

preload_models()  # load once before generating all scenes

# Optionally choose speaker / language
from bark.api import semantic_to_waveform, generate_text_semantic
speaker = "v2/en_speaker_9"

semantic_tokens = generate_text_semantic(scene.dialogue, history_prompt=speaker)
audio_array = semantic_to_waveform(semantic_tokens, history_prompt=speaker)

def generate_audio(scene: Scene, out_dir: Path):
    """
    Generate expressive speech using Bark (open source).
    Audio is automatically padded or truncated to match scene.duration.
    """
    import numpy as np, soundfile as sf, torch
    from bark import generate_audio as bark_generate
    from bark.generation import SAMPLE_RATE

    out = out_dir / f"{scene.id}.wav"
    text = f"{scene.audio_summary}. {scene.dialogue}"
    print(f"[BARK] generating audio for scene {scene.id}")

    # --- Bark synthesis ---
    audio_array = bark_generate(text)
    audio = np.array(audio_array, dtype=np.float32)

    # --- Optional: fit Bark audio to scene.duration ---
    actual_dur = len(audio) / SAMPLE_RATE
    target_len = int(scene.duration * SAMPLE_RATE)
    if actual_dur > scene.duration:
        audio = audio[:target_len]                 # truncate
    else:
        pad = target_len - len(audio)
        if pad > 0:
            audio = np.pad(audio, (0, pad))        # pad with silence

    # --- Save WAV ---
    sf.write(out, audio, SAMPLE_RATE)
    print(f"[BARK] saved {out} ({scene.duration:.2f}s target)")
    return out
