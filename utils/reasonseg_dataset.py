import os
import json
import time
import requests
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils


API_URL = "https://api.deepseek.com/chat/completions"


# ------------------------------------------------
# visualization
# ------------------------------------------------
def visualize_reasonseg_results(img_id, qa_data, args):

    img_path = os.path.join(args.image_folder, img_id + ".jpg")

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    img = np.array(Image.open(img_path).convert("RGB"))

    save_root = os.path.join(args.answer_json_dir, "visual")

    save_dir = os.path.join(save_root, img_id)

    os.makedirs(save_dir, exist_ok=True)

    for idx, qa in enumerate(qa_data):

        mask_ids = qa.get("answer_mask_ids", [])

        # question 可能是 list
        q_raw = qa.get("question", "")

        if isinstance(q_raw, list):
            question = q_raw[0]
        else:
            question = q_raw

        combined_mask = np.zeros(
            (img.shape[0], img.shape[1]),
            dtype=np.uint8
        )

        for mask_id in mask_ids:

            mask_path = os.path.join(
                args.mask_dir,
                img_id,
                f"mask_{mask_id}.png"
            )

            if not os.path.exists(mask_path):
                print(f"Mask not found: {mask_path}")
                continue

            mask = np.array(
                Image.open(mask_path).convert("L")
            )

            mask = (mask > 0).astype(np.uint8)

            if mask.shape != combined_mask.shape:
                mask = np.array(
                    Image.fromarray(mask).resize(
                        (img.shape[1], img.shape[0]),
                        Image.NEAREST
                    )
                )

            combined_mask = np.logical_or(
                combined_mask,
                mask
            )

        combined_mask = (combined_mask.astype(np.uint8) * 255)

        # 保存mask
        mask_save_path = os.path.join(
            save_dir,
            f"mask_{idx+1}.png"
        )

        Image.fromarray(combined_mask).save(mask_save_path)

        # 可视化
        plt.figure(figsize=(8, 6))

        plt.imshow(img)

        plt.imshow(
            combined_mask,
            alpha=0.5,
            cmap="jet"
        )

        plt.title(
            f"Q{idx+1}: {question}",
            fontsize=10
        )

        plt.axis("off")

        vis_save_path = os.path.join(
            save_dir,
            f"mask_{idx+1}_vis.png"
        )

        plt.savefig(
            vis_save_path,
            bbox_inches="tight"
        )

        plt.close()

    print(f"Visualization saved in {save_dir}")


# ------------------------------------------------
# mask -> RLE
# ------------------------------------------------
def binary_mask_to_rle(binary_mask):

    rle = mask_utils.encode(
        np.asfortranarray(binary_mask.astype(np.uint8))
    )

    rle["counts"] = rle["counts"].decode("utf-8")

    return rle["counts"]


# ------------------------------------------------
# 读取 description json
# ------------------------------------------------
def load_mask_description(des_path, max_masks):

    with open(des_path) as f:
        data = json.load(f)

    whole_desc = data.get("whole", {}).get("des", "")

    parts = []

    numeric_keys = [k for k in data.keys() if k.isdigit()]

    for k in sorted(numeric_keys, key=lambda x: int(x))[:max_masks]:

        parts.append({
            "mask_id": int(k),
            "description": data[k]["des"]
        })

    return whole_desc, parts


# ------------------------------------------------
# 构造 prompt
# ------------------------------------------------
def build_prompt(whole_description, parts):

    prompt_data = {
        "image_context": whole_description,
        "object_parts": parts
    }

    valid_ids = [p["mask_id"] for p in parts]

    prompt = f"""
You are an expert Dataset Architect specializing in Vision-Language Models. Your task is to generate high-quality Reasoning Segmentation (ReasonSeg) data based on the provided image context and object part descriptions.

IMAGE DATA:
{json.dumps(prompt_data, indent=2)}

Valid mask_ids in this image:
{valid_ids}

### CORE PRINCIPLE: Reasoning Depth & Knowledge-Rich Queries
Instead of simple identification, your questions must be "Implicit & Context-Heavy." 
1. Contextual Preamble: Begin with real-world knowledge.
2. Implicit Referring: Avoid naming the target object directly.
3. Always cross-reference the "whole" (global context) to verify if the object actually exists and makes sense.
4. Ensure correctness of question and answer first; difficulty can be increased later.

### NUMBER OF QUESTIONS
Generate **1–4 reasoning questions per image**.
- Prefer **quality over quantity**
- If the image is simple, **1 question is acceptable**

### TASK SPECIFICATIONS

For each reasoning point generate:

- **5 question variations**: Paraphrase the same logical query into 5 different manner. For example:
    - *Variation 1*: "When eating a bowl of soup, what kitchen tool would most likely be used to scoop and consume the soup?",
        - *Variation 2*: "Which kitchen tool is typically used to scoop and consume soup during a meal?",
        - *Variation 3*: "What is the common utensil used for consuming soup from a bowl?",
        - *Variation 4*: "What is the common kitchen utensil used for scooping and consuming soup from a bowl?",
        - *Variation 5*: "What tool would likely be utilized to scoop and consume soup while eating it from a bowl?",
        - *Variation 6*: "What kitchen tool would most likely be used to scoop and consume soup while eating it from a bowl?",
        - *Variation 7*: "What kitchen tool is typically used to scoop and consume a bowl of soup?" 
- **2–3 sentence explanation**: One detailed 2-3 sentence explanation linking the world knowledge to the visual evidence.
        - **GOOD**: "When eating a bowl of soup, a spoon would most likely be used to scoop and consume the soup. In the image, there is a spoon resting in the bowl of soup, which is placed on a dining table. The spoon is a convenient and common tool for eating soup, as it allows for easy scooping and consuming the liquid or semi-liquid food."
- **answer_labels**: List of object category names corresponding to the answer masks. For example, if the question is about "What tool would likely be utilized to scoop and consume soup while eating it from a bowl?", the answer_labels could be ["spoon"].
- **answer_mask_ids**

### QUESTION TAXONOMY (5 Categories)
    You MUST generate questions from these 5 categories. For each image, select the most appropriate categories based on the content:

    1. **Commonsense Reasoning** - Requires real-world knowledge
    - **GOOD**: "Which item suggests a birthday celebration?" (answers: birthday cake)
    - **BAD**: "What is this object?" (simple identification)

    2. **Robot Manipulation** - Needs 2+ objects for a task
    - **GOOD**: "Cut the dessert into three small portions." (answers: cake and knife in the image)
    - **GOOD**: "Identify all objects that can be used directly as payment." (answers: all the money items like coins and banknotes)

    3. **Functional Reasoning** - Groups by shared purpose, not category name.
    - **GOOD**: ""Which containers here can safely for drinking water?" (answers: cups, bottles)
    - **BAD**: "Find all cups" (too direct)

    4. **Attribute-Based Reasoning** - Based on physical properties (material, texture, or shape)
    - **GOOD**: "Which object is made of a material that can reflect the surrounding scene?" (answers: the moirror in the image)
    - **GOOD**: "Which objects are soft and can be used when sleeping?" (answers: the soft bed and pillow in the image)

    5. **Part-Based Reasoning** - Targets object components
    - Use function or affordance instead.
    - **GOOD**: "Which part of the elephant is primarily used for grasping objects?" (answer: trunk)
    - **GOOD**: "Which part of the knife should you hold to avoid getting cut?" (answer: the handle of the knife)
    - **BAD**: "Which part of the elephant is its trunk?" (too direct)


### QUALITY CONTROL

- Questions must require reasoning
- Do NOT mention mask ids in explanations
- Only use objects present in provided data
- Only use mask_ids from this list: {valid_ids}



OUTPUT JSON FORMAT:

[
{{
"reasoning_type": "Robot Manipulation / Functional Reasoning / Commonsense Reasoning / Attribute-Based / Part-Based",
"question":[
"variation1",
"variation2",
"variation3",
"variation4",
"variation5"
],
"answer_labels":["object category names"],
"answer_mask_ids":[mask_id1, mask_id2],
"explanation":"2-3 sentences explaining reasoning"
}}
]

Rules:
- Output JSON only
- Do not add extra text
"""

    return prompt


# ------------------------------------------------
# 调用 DeepSeek reasoning
# ------------------------------------------------
def call_deepseek(prompt, api_key, retry=3):

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # payload = {
    #     "model": "deepseek-reasoner",
    #     "messages": [
    #         {
    #             "role": "system",
    #             "content": "You are a visual reasoning dataset expert."
    #         },
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ],
    #     "temperature": 0.2,
    # }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a visual reasoning dataset expert."
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

            if r.status_code != 200:

                print("API error:", r.text)

                time.sleep(3)

                continue

            result = r.json()

            msg = result["choices"][0]["message"]

            content = msg.get("content", "")

            if content.strip():

                return content

        except Exception as e:

            print("Request error:", e)

        time.sleep(3)

    return None


# ------------------------------------------------
# 解析 JSON
# ------------------------------------------------
def parse_json(text):

    try:

        start = text.find("[")
        end = text.rfind("]")

        if start == -1 or end == -1:
            return None

        json_str = text[start:end+1]

        return json.loads(json_str)

    except Exception:

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

        mask = np.array(
            Image.open(mask_path).convert("L")
        ) > 0

        if merged is None:
            merged = mask
        else:
            merged = np.logical_or(merged, mask)

    if merged is None:
        return None

    return merged.astype(np.uint8)


# ------------------------------------------------
# 主函数
# ------------------------------------------------
def generate_reasonseg_dataset(args):

    json_files = [
        f for f in os.listdir(args.des_dir) if f.endswith(".json")
    ]

    print(f"Found {len(json_files)} JSON files in {args.des_dir}")

    for jf in tqdm(json_files):

        img_id = os.path.splitext(jf)[0]

        des_path = os.path.join(args.des_dir, jf)

        whole_desc, parts = load_mask_description(
            des_path,
            args.max_masks
        )

        prompt = build_prompt(whole_desc, parts)

        raw_response = call_deepseek(
            prompt,
            args.api_key
        )

        if raw_response is None:

            print("LLM failed:", img_id)

            continue

        result_json = parse_json(raw_response)

        if result_json is None:

            print("JSON parse failed:", img_id)

            continue

        final_results = []

        for item in result_json:

            mask_ids = item.get("answer_mask_ids", [])

            merged_mask = merge_masks(
                mask_ids,
                args.mask_dir,
                img_id
            )

            counts = ""

            if merged_mask is not None:
                counts = binary_mask_to_rle(merged_mask)

            item["counts"] = counts

            final_results.append(item)

        os.makedirs(args.answer_json_dir, exist_ok=True)

        save_path = os.path.join(
            args.answer_json_dir,
            img_id + ".json"
        )

        with open(save_path, "w") as f:

            json.dump(
                final_results,
                f,
                indent=2
            )

        print("Saved:", save_path)

        if args.question_visualize:
            visualize_reasonseg_results(
                img_id,
                final_results,
                args
            )