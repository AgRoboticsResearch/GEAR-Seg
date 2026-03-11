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




def build_generation_prompt(whole_description, parts_list):
    """
    Constructs the prompt for DeepSeek to generate reasoning questions based on part descriptions.
    """
    prompt_data = {
        "image_context": whole_description,
        "object_parts_and_descriptions": parts_list
    }
    parts_json_str = json.dumps(prompt_data, indent=2)

    system_instruction = """### Role Definition
    You are an expert Dataset Architect specializing in Vision-Language Models. Your task is to generate high-quality Reasoning Segmentation (ReasonSeg) data. 
    
    
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

    ### OUTPUT FORMAT (Strict JSON)
    Output ONLY a JSON list containing one object:
    [
        {
            "reasoning_type": "Commonsense Reasoning / Part-Based / Robot-Manipulation / Functional / Attribute-Based",
            "questions": [
            "Simple reasoning question with context",
            "Politer or more descriptive version",
            "Focus on the physical action/examination",
            "Complex sentence structure",
            "Inquiry about the aspect or evidence",
            "Concise but context-aware version"
            ],
            "answer labels":["label1", lable2"],
            "answer_mask_ids": [integer_ids],
            "explanation": "Start with world knowledge, then point to visual evidence, and conclude with the target identification."
        }
    ]"""

    user_content = f"""IMAGE DATA
    {parts_json_str}

    ### GENERATION INSTRUCTIONS (STRICT)
    Generate EXACTLY 1 high-quality question. Generate more questions ONLY if absolutely necessary.

    IMPORTANT INSTRUCTIONS:
    1. Prioritize question quality over quantity, you can generate only one qyestion if the image is simple;
    2. Each question must require genuine reasoning (not simple identification);
    3. Ensure answers are justifiable from the provided descriptions;
    4. Include a mix of single-mask and multi-mask questions when possible;
    5. For part-based questions, DO NOT mention the part name directly in the question.

    Output ONLY valid JSON with no additional text.
    """

    return system_instruction, user_content


# ------------------------------------------------
# mask -> RLE
# ------------------------------------------------
def binary_mask_to_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle["counts"]


# ------------------------------------------------
# 读取 description
# ------------------------------------------------
def load_mask_description(des_path, max_masks):

    with open(des_path) as f:
        data = json.load(f)

    whole_desc = data.get("whole", {}).get("des", "")

    mask_desc = {}

    numeric_keys = [k for k in data.keys() if k.isdigit()]

    for k in sorted(numeric_keys, key=lambda x: int(x))[:max_masks]:
        mask_desc[k] = data[k]["des"]

    return whole_desc, mask_desc





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

            # print("\n----- reasoning -----\n")
            # print(reasoning[:800])

            # print("\n----- answer -----\n")
            # print(content)

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
def generate_reasonseg_dataset(img_id, args):


    des_path = os.path.join(args.des_dir, img_id + ".json")
    img_path = os.path.join(args.image_dir, img_id + ".jpg")



    whole_desc, mask_desc = load_mask_description(
        des_path, args.max_masks
    )

    sys_prompt, user_prompt = build_generation_prompt(whole_desc, mask_desc)
    raw_response = call_deepseek(sys_prompt, user_prompt)




    # print("LLM answer:", answer)

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



