#!/usr/bin/env python3
"""
timed_script_video_pipeline.py
Loads a multi-scene script JSON, calls wan2_workflow for each scene.
"""

import json
from pathlib import Path
from wan2_workflow import process_script, CONFIG

def run_script(script_path):
    with open(script_path) as f:
        data = json.load(f)
    print(f"🎥 Running script: {script_path}")
    process_script(data)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python timed_script_video_pipeline.py script.json")
        exit(1)
    run_script(sys.argv[1])
