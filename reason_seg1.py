import os
import json
import time
import argparse
import requests
import numpy as np
import re

from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils


API_URL = "https://api.deepseek.com/chat/completions"


# ------------------------------------------------
# mask -> RLE
# ------------------------------------------------
def binary_mask_to_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle["counts"]


# ------------------------------------------------
# 读取 mask description（不读取 whole）
# ------------------------------------------------
def load_mask_description(des_path, max_masks):

    with open(des_path) as f:
        data = json.load(f)

    mask_desc = {}

    numeric_keys = [k for k in data.keys() if k.isdigit()]

    for k in sorted(numeric_keys, key=lambda x: int(x))[:max_masks]:
        mask_desc[k] = data[k]["des"]

    return mask_desc


# ------------------------------------------------
# Prompt（只基于 mask description）
# ------------------------------------------------
def build_reasonseg_prompt(question, mask_desc):

    mask_text = ""

    for mid, desc in mask_desc.items():
        mask_text += f"[Mask {mid}]: {desc}\n"

    prompt = f"""
    You are an expert visual reasoning assistant for segmentation tasks.

    Goal:
    Determine which mask regions correspond to the object mentioned in the question.

    You are given ONLY descriptions of segmented regions.  
    There is NO global scene description.

    Reasoning Steps:

    1. Carefully read the question.
    2. Analyze the semantic meaning of each mask description.
    3. Use commonsense reasoning about object functions and attributes.
    4. Select masks referring to the SAME object category.
    5. Return at most 6 masks.
    6. If nothing matches return [].

    Question:
    {question}

    Mask Regions:
    {mask_text}

    Output format (STRICT JSON ONLY):

    {{
    "mask_ids": ["id1","id2"],
    "explanation": "short reasoning"
    }}

    Do NOT output anything else.
    """

    return prompt


# ------------------------------------------------
# JSON 解析
# ------------------------------------------------
def parse_json(text):

    if text is None:
        return None

    match = re.search(r'\{[\s\S]*\}', text)

    if match:
        try:
            return json.loads(match.group())
        except:
            return None

    return None


# ------------------------------------------------
# DeepSeek API
# ------------------------------------------------
def call_api(prompt, api_key, retry=3):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a visual reasoning expert for segmentation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "thinking": {"type": "enabled"}
    }

    for i in range(retry):

        try:

            r = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            print("HTTP status:", r.status_code)

            if r.status_code != 200:
                print("API error:", r.text)
                time.sleep(3)
                continue

            result = r.json()

            msg = result["choices"][0]["message"]

            reasoning = msg.get("reasoning_content", "")
            content = msg.get("content", "")

            if content.strip():
                return content

            if reasoning.strip():
                return reasoning

        except Exception as e:

            print("Request error:", e)

        time.sleep(3)

    return None


# ------------------------------------------------
# merge masks
# ------------------------------------------------
def merge_masks(mask_ids, mask_root, image_id):

    merged = None

    mask_dir = os.path.join(mask_root, image_id)

    if not os.path.exists(mask_dir):
        return None

    for mid in mask_ids:

        mask_path = os.path.join(mask_dir, f"mask_{mid}.png")

        if not os.path.exists(mask_path):
            continue

        mask = np.array(Image.open(mask_path).convert("L")) > 0

        if merged is None:
            merged = mask
        else:
            merged = np.logical_or(merged, mask)

    if merged is None:
        return None

    return merged.astype(np.uint8)


# ------------------------------------------------
# visualize
# ------------------------------------------------
def visualize(image_path, mask, question, save_path):

    img = np.array(Image.open(image_path).convert("RGB"))

    vis = img.copy()

    vis[mask > 0] = (
        vis[mask > 0] * 0.65 + np.array([255, 0, 0]) * 0.35
    ).astype(np.uint8)

    vis_img = Image.fromarray(vis)

    H, W = img.shape[:2]

    border = 20

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    tmp = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(tmp)

    bbox = draw.textbbox((0, 0), question, font=font)

    header = bbox[3] - bbox[1] + border * 2

    canvas = Image.new(
        "RGB",
        (W + border * 2, H + header + border),
        (255, 255, 255),
    )

    canvas.paste(vis_img, (border, header))

    draw = ImageDraw.Draw(canvas)

    draw.text((border, border), question, (0, 0, 0), font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    canvas.save(save_path)


# ------------------------------------------------
# 单图处理
# ------------------------------------------------
def process_image(img_id, args):

    ques_path = os.path.join(args.ques_dir, img_id + ".json")
    des_path = os.path.join(args.des_dir, img_id + ".json")
    img_path = os.path.join(args.image_dir, img_id + ".jpg")

    if not os.path.exists(ques_path) or not os.path.exists(des_path):
        return

    with open(ques_path) as f:
        ques = json.load(f)

    question = ques["text"][0]

    mask_desc = load_mask_description(
        des_path, args.max_masks
    )

    prompt = build_reasonseg_prompt(
        question,
        mask_desc
    )

    answer = call_api(prompt, args.api_key)

    print("LLM answer:", answer)

    result_json = parse_json(answer)

    if result_json is None:
        print("JSON parse failed")
        return

    selected_ids = result_json.get("mask_ids", [])
    explanation = result_json.get("explanation", "")

    merged_mask = merge_masks(
        selected_ids,
        args.mask_dir,
        img_id
    )

    counts = None

    if merged_mask is not None:
        counts = binary_mask_to_rle(merged_mask)

    result = {
        "text": [question],
        "image_name": img_id + ".jpg",
        "answer_mask_id": selected_ids,
        "explanation": explanation,
        "shapes": [
            {
                "label": "target",
                "labels": ["target"],
                "shape_type": "mask",
                "counts": counts,
            }
        ],
    }

    os.makedirs(args.answer_json_dir, exist_ok=True)

    out_json = os.path.join(args.answer_json_dir, img_id + ".json")

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved:", out_json)

    if merged_mask is not None:

        save_vis = os.path.join(
            args.answer_visual_dir,
            img_id + ".png"
        )

        visualize(
            img_path,
            merged_mask,
            question,
            save_vis
        )


# ------------------------------------------------
# main
# ------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ques_dir", required=True)
    parser.add_argument("--des_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)

    parser.add_argument("--answer_json_dir", required=True)
    parser.add_argument("--answer_visual_dir", required=True)

    parser.add_argument("--api_key", required=True)

    parser.add_argument("--max_masks", type=int, default=130)

    args = parser.parse_args()

    os.makedirs(args.answer_json_dir, exist_ok=True)
    os.makedirs(args.answer_visual_dir, exist_ok=True)

    files = os.listdir(args.ques_dir)

    for f in files:

        if not f.endswith(".json"):
            continue

        img_id = os.path.splitext(f)[0]

        print("\nProcessing:", img_id)

        process_image(img_id, args)


if __name__ == "__main__":
    main()



# python /home/nya/code/code/2-Reasoning-Seg/GEAR-Seg/GEAR-Seg/reason_seg1.py --ques_dir /home/nya/code/code/2-Reasoning-Seg/GEAR-Seg/GEAR-Seg/img/ReasonSeg/question/val --des_dir /home/nya/code/code/2-Reasoning-Seg/GEAR-Seg/GEAR-Seg/output/descriptions/val_whole --image_dir /home/nya/code/code/2-Reasoning-Seg/GEAR-Seg/GEAR-Seg/img/ReasonSeg/img/val --mask_dir /home/nya/code/code/2-Reasoning-Seg/GEAR-Seg/GEAR-Seg/output/masks/val --answer_json_dir /home/nya/code/code/2-Reasoning-Seg/GEAR-Seg/GEAR-Seg/output/answers_part/json --answer_visual_dir /home/nya/code/code/2-Reasoning-Seg/GEAR-Seg/GEAR-Seg/output/answers_part/visual --api_key "sk-bbccc192438e443798f779f20da36d4b"