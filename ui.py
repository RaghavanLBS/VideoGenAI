import gradio as gr, json
from wan2_workflow import CONFIG, process_script

def run_json_script(script_json, width, height, frames, fp16, memory_saver):
    CONFIG["default_width"] = width
    CONFIG["default_height"] = height
    CONFIG["default_frames"] = frames
    CONFIG["fp16"] = fp16
    CONFIG["memory_saver"] = memory_saver
    CONFIG["gpu_name"] = "A40"
    out_path = process_script(json.loads(script_json))
    return str(out_path)

with gr.Blocks(title="WAN 2.2 Movie Builder") as app:
    gr.Markdown("## 🎬 WAN 2.2 + Bark — Script-Driven Movie Studio")

    with gr.Row():
        width = gr.Slider(256,1024,512,step=64,label="Width")
        height = gr.Slider(256,1024,512,step=64,label="Height")
        frames = gr.Slider(16,96,64,step=8,label="Frames")
    fp16 = gr.Checkbox(True,label="Use FP16 (recommended for A40)")
    memory_saver = gr.Checkbox(True,label="Enable Memory Saver")

    script_json = gr.Textbox(lines=20,label="JSON Script",value=json.dumps({
        "scenes": [
            {
                "id":"scene1",
                "summary":"Dog and cat walking through a sunset field",
                "audio_summary":"soft wind and gentle tone",
                "dialogue":"Dog: What a beautiful day! Cat: Yes, perfect for a walk.",
                "camera":"slow pan right, warm light",
                "duration":6.0
            },
            {
                "id":"scene2",
                "summary":"Cat jumps onto rock, dog laughs",
                "audio_summary":"light laughter, playful tone",
                "dialogue":"Cat: Look, I can climb higher! Dog: Don’t fall!",
                "camera":"tracking shot",
                "duration":5.0
            }
        ]
    },indent=2))
    run_btn = gr.Button("🚀 Generate Full Movie")
    out_video = gr.Video(label="Final Movie Output")

    run_btn.click(fn=run_json_script,
                  inputs=[script_json,width,height,frames,fp16,memory_saver],
                  outputs=[out_video])

app.launch(server_name="0.0.0.0",server_port=7860)
