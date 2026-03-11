import os
import json
import requests
from tqdm import tqdm

import subprocess
import time
import requests


def close_ollama(ollama_process):
    if ollama_process is not None:
        print("Stopping Ollama server...")
        ollama_process.terminate()


def ensure_ollama_running(ollama_bin):

    url = "http://localhost:11434"

    try:
        requests.get(url, timeout=2)
        print("Ollama server already running.")
        return None
    except:
        print("Ollama server not running. Starting...")

    ollama_process = subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待 server ready
    for _ in range(30):

        try:
            requests.get(url, timeout=2)
            print("Ollama server started.")
            return ollama_process
        except:
            time.sleep(1)

    raise RuntimeError("Failed to start Ollama server.")


PROMPT = (
    "Describe the image in one concise paragraph. "
    "Mention only the main objects and their spatial relationships. "
    "Do not add speculation. "
    "Do not count objects."
)


OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_whole_descriptions(
        args,
        des_dir,
        model_name="llama3.2-vision:latest"
):



    json_files = [
        f for f in os.listdir(des_dir) if f.endswith(".json")
    ]

    print(f"Found {len(json_files)} JSON files in {des_dir}")

    for jf in tqdm(json_files):

        json_path = os.path.join(des_dir, jf)

        with open(json_path, "r") as f:
            print(f"Processing {jf}...")
            data = json.load(f)

        # 已存在 whole
        if "whole" in data:
            print(f"Skipping {jf}, 'whole' already exists.")
            continue

        image_name = os.path.basename(json_path).replace(".json", ".jpg")
        img_path = os.path.join(args.image_folder, image_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        message = json.dumps([
            {
                "role": "user",
                "content": PROMPT,
                "images": [img_path]
            }
        ])

        try:
            result = subprocess.run(
                [args.ollama_path, "run", model_name],
                input=message,
                capture_output=True,
                text=True,
                timeout=120
            )

            output = result.stdout.strip()

            if not output:
                print(f"No output from Ollama for {jf}")
                continue

            # 插入 whole description 到 JSON 最前面
            data = {"whole": {"des": output}, **data}

            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Successfully updated {jf} with whole description.")

        except subprocess.TimeoutExpired:
            print(f"Timeout while processing {jf}")
        except Exception as e:
            print(f"Error processing {jf}: {e}")

    print("Whole description generation finished.")