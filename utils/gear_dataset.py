import os
import json
import time
import requests
import numpy as np
import re
from tqdm import tqdm

from PIL import Image
from pycocotools import mask as mask_utils


API_URL = "https://api.deepseek.com/chat/completions"


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

    prompt = f"""
You are an expert Dataset Architect specializing in Vision-Language Models. Your task is to generate high-quality Reasoning Segmentation (ReasonSeg) data. 

IMAGE DATA:
{json.dumps(prompt_data, indent=2)}

### CORE PRINCIPLE: Reasoning Depth & Knowledge-Rich Queries
Instead of simple identification, your questions must be "Implicit & Context-Heavy." 
1. **Contextual Preamble**: Every reasoning point must start with a brief sentence about real-world knowledge or logic before asking the question.
2. **Implicit Referring**: Avoid naming the target object directly. Use its function, state, or relationship to the context.

### TASK SPECIFICATIONS
For a given image, you must identify ONE core reasoning point and generate:
- **6 Variations of the Question**: Paraphrase the same logical query into 6 different manner. For example:
    - *Variation 1*: "When eating a bowl of soup, what kitchen tool would most likely be used to scoop and consume the soup?",
    - *Variation 2*: "Which kitchen tool is typically used to scoop and consume soup during a meal?",
    - *Variation 3*: "What is the common utensil used for consuming soup from a bowl?",
    - *Variation 4*: "What is the common kitchen utensil used for scooping and consuming soup from a bowl?",
    - *Variation 5*: "What tool would likely be utilized to scoop and consume soup while eating it from a bowl?",
    - *Variation 6*: "What kitchen tool would most likely be used to scoop and consume soup while eating it from a bowl?",
    - *Variation 7*: "What kitchen tool is typically used to scoop and consume a bowl of soup?"
- **Expanded Explanation**: One detailed 2-3 sentence explanation linking the world knowledge to the visual evidence.
        - **GOOD**: "When eating a bowl of soup, a spoon would most likely be used to scoop and consume the soup. In the image, there is a spoon resting in the bowl of soup, which is placed on a dining table. The spoon is a convenient and common tool for eating soup, as it allows for easy scooping and consuming the liquid or semi-liquid food."
    -**The label of targets asked by question**: Summarize the labels of the items asked about in the question.
        - **GOOD**: ["spoon"]
        
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

### QUALITY CONTROL GUIDELINES

#### 1. REASONING DEPTH REQUIREMENTS
Each question must require at least one of:
- **Functional Inference**: Understanding how objects are used
- **Property Deduction**: Inferring attributes from descriptions
- **Relational Logic**: Understanding object relationships
- **World Knowledge**: Applying commonsense beyond the image

#### 2. EXPLANATION STANDARDS
- **Reasoning Chain**: Clearly articulate the logical steps.
- **Property Reference**: Reference object properties from descriptions.
- **No Mask IDs**: Never mention mask IDs in explanations.
- **Function Focus**: Emphasize functional or property-based reasoning.

#### 3. **No Hallucination**:
- Only reference objects explicitly present in provided data.
- Do not infer scene context, time, or user intent not supported by descriptions.

OUTPUT JSON FORMAT:

[
{
"reasoning_type": "Robot Manipulation / Functional Reasoning / Commonsense Reasoning / Attribute-Based / Part-Based",

"question":[
"variation1",
"variation2",
"variation3",
"variation4",
"variation5"
],

"answer_labels":["object category names"],

"answer_mask_ids":[mask ids],

"explanation":"2-3 sentences explaining reasoning"
},
....
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

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a visual reasoning expert."
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

        s = text.find("[")
        e = text.rfind("]")

        if s == -1 or e == -1:
            return None

        return json.loads(text[s:e + 1])

    except:
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
# 主函数（供主程序 import）
# ------------------------------------------------
def generate_reasonseg_dataset(args,des_dir):

    json_files = [
        f for f in os.listdir(des_dir) if f.endswith(".json")
    ]

    print(f"Found {len(json_files)} JSON files in {des_dir}")

    for jf in tqdm(json_files):
        des_path = os.path.join(des_dir, jf)

        whole_desc, parts = load_mask_description(
            des_path,
            args.max_masks
        )

    prompt = build_prompt(whole_desc, parts)

    raw_response = call_deepseek(prompt, args.api_key)

    if raw_response is None:
        print("LLM failed")
        return

    result_json = parse_json(raw_response)

    if result_json is None:
        print("JSON parse failed")
        return

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