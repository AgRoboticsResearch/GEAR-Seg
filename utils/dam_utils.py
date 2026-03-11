import os
import numpy as np
from PIL import Image
from dam import DescribeAnythingModel, disable_torch_init
import torch
import json
import re



def find_mask_bbox(mask_np):
    """Find bounding box of white area in mask"""
    white_pixels = np.where(mask_np > 0)
    if len(white_pixels[0]) == 0:
        return None
    min_row, max_row = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_col, max_col = np.min(white_pixels[1]), np.max(white_pixels[1])
    return min_row, min_col, max_row, max_col

def expand_bbox(bbox, height, width, scale):
    min_row, min_col, max_row, max_col = bbox
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    bbox_height = max_row - min_row + 1
    bbox_width = max_col - min_col + 1
    new_height = int(bbox_height * scale)
    new_width = int(bbox_width * scale)
    new_min_row = max(0, center_row - new_height // 2)
    new_min_col = max(0, center_col - new_width // 2)
    new_max_row = min(height - 1, center_row + new_height // 2)
    new_max_col = min(width - 1, center_col + new_width // 2)
    return new_min_row, new_min_col, new_max_row, new_max_col



def init_dam_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disable_torch_init()
    return DescribeAnythingModel(
        model_path=args.dam_model_path,
        conv_mode="v1",
        prompt_mode="full+focal_crop"
    ).to(device)

def get_mask_description(dam_model, img_path, mask_path, args):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != img.size:
        img = img.resize(mask.size, Image.LANCZOS)
    img_np = np.array(img)
    mask_np = np.array(mask)
    height, width = mask_np.shape
    bbox = find_mask_bbox(mask_np)
    if bbox is None:
        return "No object found in mask."
    expanded_bbox = expand_bbox(bbox, height, width, args.scale)
    min_row, min_col, max_row, max_col = expanded_bbox
    cropped_img_np = img_np[min_row:max_row + 1, min_col:max_col + 1]
    cropped_mask_np = mask_np[min_row:max_row + 1, min_col:max_col + 1]
    cropped_img = Image.fromarray(cropped_img_np)
    cropped_mask = Image.fromarray(cropped_mask_np)
    
    query = "<image>\nDescribe the masked region in detail."
    description = dam_model.get_description(
            cropped_img, cropped_mask, query,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens
        )
    return description


def save_description(dam_model, masks_dir, subfolder_descriptions_output, args):
    """
    Generate DAM descriptions for all masks.
    Each image will have its own JSON file.
    """


    for img_name in sorted(os.listdir(masks_dir)):

        mask_dir = os.path.join(masks_dir, img_name)
        if not os.path.isdir(mask_dir):
            continue

        json_path = os.path.join(subfolder_descriptions_output, f"{img_name}.json")

        # -----------------------------
        # Load existing results
        # -----------------------------
        results = {}

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                print(f"Loaded existing JSON: {json_path}")
            except Exception as e:
                print(f"Failed to read {json_path}: {e}")
                results = {}

        # -----------------------------
        # Collect mask files
        # -----------------------------
        mask_files = sorted(
            f for f in os.listdir(mask_dir)
            if f.startswith("mask_") and f.endswith(".png")
        )

        mask_indices = [
            re.match(r"mask_(\d+)", f).group(1)
            for f in mask_files
        ]

        # -----------------------------
        # Check completion
        # -----------------------------
        if set(results.keys()) == set(mask_indices):
            print(f"✓ {img_name} already complete, skip")
            continue
        else:
            print(
                f"Processing {img_name} "
                f"(json:{len(results)} mask:{len(mask_indices)})"
            )

        # -----------------------------
        # Find image path
        # -----------------------------
        img_path = None

        

        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:

            candidate = os.path.join(args.image_folder, f"{img_name}{ext}")

            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"Image not found for {img_name}")
            continue

        # -----------------------------
        # Process masks
        # -----------------------------
        for mask_file in mask_files:

            mask_idx = re.match(r"mask_(\d+)", mask_file).group(1)

            if mask_idx in results:
                continue

            mask_path = os.path.join(mask_dir, mask_file)

            try:

                description = get_mask_description(
                    dam_model,
                    img_path,
                    mask_path,
                    args
                )

                results[mask_idx] = {
                    "des": description
                    # "filename": mask_path
                }

            except Exception as e:
                print(f"Error processing {mask_file}: {e}")
                continue

        # -----------------------------
        # Sort by mask index
        # -----------------------------
        try:
            results = dict(sorted(
                (item for item in results.items() if item[0].isdigit()),
                key=lambda x: int(x[0])
            ))
        except ValueError as e:
            print(f"Skipping non-integer keys during sorting: {e}")

        # -----------------------------
        # Save JSON
        # -----------------------------
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved {img_name} results → {json_path}")