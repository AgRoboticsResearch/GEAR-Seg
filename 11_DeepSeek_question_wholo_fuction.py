import requests
import json
import os
import time
import datetime
import pytz
import argparse
import re

# ---------------------- Configuration ----------------------
API_KEY = "sk-bbccc192438e443798f779f20da36d4b"
API_URL = "https://api.deepseek.com/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# ---------------------- Wait Function ----------------------
def wait_until_target_time(target_time_str, timezone='Asia/Shanghai'):
    tz = pytz.timezone(timezone)
    try:
        target_dt = tz.localize(datetime.datetime.strptime(target_time_str, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        print("❌ Invalid time format. Use 'YYYY-MM-DD HH:MM:SS'")
        return

    now = datetime.datetime.now(tz)
    wait_seconds = (target_dt - now).total_seconds()
    if wait_seconds <= 0:
        print("Target time has passed. Starting immediately.")
        return

    print(f"Waiting until {target_time_str} ({wait_seconds:.2f} seconds)...")
    while wait_seconds > 600:
        print(f"Still waiting... {wait_seconds/3600:.2f} hours left.")
        time.sleep(600)
        now = datetime.datetime.now(tz)
        wait_seconds = (target_dt - now).total_seconds()

    if wait_seconds > 0:
        time.sleep(wait_seconds)
    print("Time reached! Starting process.")

# ---------------------- Prompt Construction ----------------------
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
# ---------------------- DeepSeek API Call ----------------------
def call_deepseek(system_prompt, user_prompt, max_retries=3):
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8,
        "response_format": {"type": "json_object"},
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(data), timeout=120)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return content
            else:
                print(f"❌ API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"❌ Request failed (Attempt {attempt+1}/{max_retries}): {e}")
        time.sleep(3)
    return None

# ---------------------- Processing Logic ----------------------
def process_reasonseg(val_no_filename_path, output_dir):
    if not os.path.exists(val_no_filename_path):
        print(f"❌ Input file not found: {val_no_filename_path}")
        return

    with open(val_no_filename_path, "r", encoding='utf-8') as f:
        val_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    total_images = len(val_data)
    print(f"🚀 Starting processing for {total_images} images...")

    for i, (img_key, masks) in enumerate(val_data.items()):
        # if i<100:
        #     continue
        if True:
            img_name = img_key.replace("train4/", "")
            print(f"\n[{i+1}/{total_images}] 🖼 Processing {img_key}")
            out_path = os.path.join(output_dir, f"{img_name}.json")
            if os.path.exists(out_path):
                print(f"⏭️  Output already exists for {img_name}, skipping.")
                continue

            # ---------------- Prepare Data ----------------
            whole_description = masks.get("whole", {}).get("des", "")
            mask_keys = [k for k in masks.keys() if k != "whole"]
            try:
                sorted_keys = sorted(mask_keys, key=lambda x: int(x))
            except ValueError:
                sorted_keys = sorted(mask_keys)

            parts_list = []
            for key in sorted_keys[:]:  # limit masks
                description = masks[key].get("des", "No description")
                try:
                    m_id = int(key)
                except ValueError:
                    m_id = key
                parts_list.append({"mask_id": m_id, "description": description})

            if not parts_list:
                print("⚠️ No masks found for this image.")
                continue

            # ---------------- Build Prompt ----------------
            sys_prompt, user_prompt = build_generation_prompt(whole_description, parts_list)

            # ---------------- Call API ----------------
            raw_response = call_deepseek(sys_prompt, user_prompt)
            if raw_response is None:
                print("❌ Failed to get response from API. Skipping.")
                continue

            # ---------------- Parse & Save JSON ----------------
            try:
                json_str = raw_response.strip()
                if "```json" in json_str:
                    json_str = re.search(r'```json\s*([\s\S]*?)\s*```', json_str).group(1)
                elif "```" in json_str:
                    json_str = re.search(r'```\s*([\s\S]*?)\s*```', json_str).group(1)

                generated_qa = json.loads(json_str)
                if isinstance(generated_qa, dict):
                    if "questions" in generated_qa:
                        generated_qa = generated_qa["questions"]
                    elif "pairs" in generated_qa:
                        generated_qa = generated_qa["pairs"]
                    else:
                        generated_qa = [generated_qa]
                if not isinstance(generated_qa, list):
                    print(f"⚠️ API returned invalid structure: {type(generated_qa)}")
                    continue
            except Exception as e:
                print(f"❌ JSON Parse Error: {e}")
                print(f"Raw output (first 500 chars): {raw_response[:500]}")
                continue

            with open(out_path, "w", encoding='utf-8') as f:
                json.dump(generated_qa, f, indent=4, ensure_ascii=False)

            print(f"✅ Generated {len(generated_qa)} Q&A pairs. Saved to: {out_path}")
            time.sleep(0.5)

    print("\n🎉 All processing complete.")

# ---------------------- Main Execution ----------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Reasoning Segmentation Dataset via DeepSeek API")
    parser.add_argument("--input_path", 
                        default="/mnt/nas/fruit_dataset/wyn/cs_dataset/SDR-ReasonSeg-Data/LVIS_train5000/DAM/LVIS-5000/train4/train4_with_whole.json",
                        help="Path to the input JSON file containing mask descriptions.")
    parser.add_argument("--out_dir", 
                        default="/mnt/nas/fruit_dataset/wyn/cs_dataset/SDR-ReasonSeg-Data/LVIS_train5000/GPT-Ques/train4_new_prompt6",
                        help="Directory to save generated JSONs.")
    parser.add_argument("--run_at", default=None, 
                        help="Schedule run at specific time (YYYY-MM-DD HH:MM:SS)")
    args = parser.parse_args()
# /mnt/nas/fruit_dataset/wyn/cs_dataset/SDR-ReasonSeg-Data/LVIS_train5000/DAM/LVIS-5000/train5/train5_with_whole.json
# /mnt/nas/fruit_dataset/wyn/cs_dataset/SDR-ReasonSeg-Data/LVIS_trai5000/DAM/LVIS-5000/train5/train5_with_whole.json
    if args.run_at:
        wait_until_target_time(args.run_at)

    if not os.path.exists(args.input_path):
        print(f"❌ Error: Input path does not exist: {args.input_path}")
        return

    process_reasonseg(
        val_no_filename_path=args.input_path,
        output_dir=args.out_dir
    )

if __name__ == "__main__":
    main()
