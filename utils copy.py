import argparse
import sys
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dam import DescribeAnythingModel, disable_torch_init
from sentence_transformers import SentenceTransformer
import cv2
from scipy.ndimage import label as label_region
import re
import json
import matplotlib.pyplot as plt
import copy
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator



def build_output_dirs(output_root, split):

    base = os.path.join(output_root, split)

    dirs = {
        "sam_mask": os.path.join(base, "sam", "masks"),
        "sam_vis": os.path.join(base, "sam", "anns_visual"),
        "des": os.path.join(base, "dam", "descriptions"),
        "ans_json": os.path.join(base, "answers", "json"),
        "ans_vis": os.path.join(base, "answers", "visual"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs



def show_anns(sorted_anns, image, anns_imgname_dir, img_name, borders=True):
    if len(sorted_anns) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=3)

        # 标注掩码的索引
        y, x = np.mean(np.argwhere(m), axis=0).astype(int)
        ax.text(x, y, str(i), color='white', fontsize=15, ha='center', va='center', weight='bold')

    ax.imshow(img)
    plt.axis('off')
    save_path = os.path.join(anns_imgname_dir, f'{img_name}.png')
    plt.savefig(save_path)
    plt.close(fig)

def save_mask(anns, imgname_dir):
    os.makedirs(imgname_dir, exist_ok=True)

    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for i, ann in enumerate(anns):
        #a = ann['original_index']
        mask = ann['segmentation']
        mask = np.stack([mask]*3, axis=-1)   #如果不进行remove处理，这句不用注释

        img = (mask*255).astype(np.uint8)  # Setting mask as white
        cv2.imwrite(f'{imgname_dir}/mask_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def filter_masks_by_overlap(masks, threshold):
    if np.__version__ >= '1.20':
        bool_type = np.bool_  # 或 np.bool_
    else:
        bool_type = np.bool
    masks_np = [np.array(mask['segmentation'], dtype=bool_type) for mask in masks]
    areas = [np.sum(mask) for mask in masks_np]
    keep = torch.ones(len(masks_np), dtype=torch.bool)
    scores = [mask['stability_score'] for mask in masks]
    keep = torch.ones(len(masks_np), dtype=torch.bool)

    # 遍历每个掩码
    for i in range(len(masks_np)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(masks_np)):
            if not keep[j]:
                continue
            
            # 计算交集和 IoU
            intersection = np.logical_and(masks_np[i], masks_np[j]).astype(np.float32).sum()
            smaller_area = min(areas[i], areas[j])
            if intersection > threshold * smaller_area:
                if scores[i] < scores[j]:
                    keep[i] = False
                else:
                    keep[j] = False

    # 过滤后的掩码
    filtered_masks = [mask for idx, mask in enumerate(masks) if keep[idx]]
    
    return filtered_masks


def init_models(args):
    """Initialize DAM, Sentence Transformer, and SAM models"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    

    disable_torch_init()
    
    # Initialize DAM
    dam = DescribeAnythingModel(
        model_path=args.dam_model_path,
        conv_mode="v1",
        prompt_mode="full+focal_crop"
    ).to(device)
    
    # Initialize Sentence Transformer
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    # 主动清理内存
    torch.cuda.empty_cache()
    # Initialize SAM
    sam_model = build_sam2(args.model_cfg, args.sam_model_path, device=device)
    sam_predictor = SAM2AutomaticMaskGenerator(sam_model, 
                                               points_per_side=args.points_per_side,
                                               min_mask_region_area=args.min_mask_region_area, 
                                               crop_n_layers=args.crop_n_layers, 
                                               stability_score_offset=args.stability_score_offset)
    
    return dam, sam_predictor




def get_mask_description(dam_model, img_path, mask_path, args):
    """Get description for masked region"""

    # Load image and mask
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Ensure image and mask size match
    if mask.size != img.size:
        img = img.resize(mask.size, Image.LANCZOS)

    # Convert to numpy
    img_np = np.array(img)
    mask_np = np.array(mask)

    height, width = mask_np.shape

    # Find mask bbox
    bbox = find_mask_bbox(mask_np)

    if bbox is None:
        print("No white region found in mask.")
        return "No object found in mask."

    # Expand bbox
    expanded_bbox = expand_bbox(bbox, height, width, args.scale)

    min_row, min_col, max_row, max_col = expanded_bbox

    # Crop image and mask (保持原始分辨率)
    cropped_img_np = img_np[min_row:max_row + 1, min_col:max_col + 1]
    cropped_mask_np = mask_np[min_row:max_row + 1, min_col:max_col + 1]

    # Convert back to PIL
    cropped_img = Image.fromarray(cropped_img_np)
    cropped_mask = Image.fromarray(cropped_mask_np)

    # Generate description
    query = "<image>\nDescribe the masked region in detail."

    description = dam_model.get_description(
        cropped_img,
        cropped_mask,
        query,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens
    )

    return description




def generate_all_sam_mask(args, sam_predictor, image_folder, masks_output_folder, anns_imgname_dir):
    """
    Generates and saves SAM masks for all images in a folder.
    """
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        #image_path = 'xxx/xx.png'
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        img_name = os.path.splitext(image_file)[0]
        mask_imgname_dir = os.path.join(masks_output_folder, img_name)

        
        # Use default point grid for automatic mask generation
        masks2 = sam_predictor.generate(image)

        sorted_anns = sorted(masks2, key=(lambda x: x['area']), reverse=True)
        if args.enable_mask_nms:
            sorted_anns = filter_masks_by_overlap(sorted_anns, args.mask_nms_thresh)
            save_mask(sorted_anns, mask_imgname_dir)
        else:
            save_mask(sorted_anns, mask_imgname_dir)

        if args.save_anns:
            show_anns(sorted_anns, image, anns_imgname_dir, img_name)
        
        
        del masks2, sorted_anns
        torch.cuda.empty_cache()
        print(f"Successfully save masks to {mask_imgname_dir}.")
    print(f"Finished generating masks for {len(image_files)} images.")






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


def get_mask_edge_points(mask, img_width, img_height, points_per_contour):
    """Get edge points from mask
    Args:
        mask: Binary mask image
        img_width: Image width
        img_height: Image height
        points_per_contour: Number of points to sample per contour
    Returns:
        edge_points_str: Edge point coordinates string in format "x1 y1 x2 y2 ..."
    """
    labelled_mask, num_labels = label_region(mask)
    edge_points_str = ""
    
    for region_label in range(1, num_labels + 1):
        # Extract current region mask
        mask_cur = ((labelled_mask == region_label) * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_cur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
            
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        c = c.reshape(-1, 2)
        
        # Sample contour points
        num_points = len(c)
        skip = max(1, num_points // points_per_contour)
        approx_sparse = c[::skip]
        
        # Find bottom point as starting point
        bottom_point_index = np.argmax(approx_sparse[:, 1])
        
        # Reorder points starting from bottom
        sorted_points = np.concatenate([approx_sparse[bottom_point_index:], approx_sparse[:bottom_point_index]])
        
        # Convert to normalized coordinates and build string
        edge_points_str += ' ' + ' '.join(
            f'{format(point[0]/img_width, ".6f")} {format(point[1]/img_height, ".6f")}' 
            for point in sorted_points
        )
    
    return edge_points_str.strip()

import os
import json
import re


def save_description(dam_model, subfolder_masks_output, subfolder_descriptions_output, args):
    """
    Generate DAM descriptions for all masks.
    Each image will have its own JSON file.
    """

    os.makedirs(subfolder_descriptions_output, exist_ok=True)

    for img_name in sorted(os.listdir(subfolder_masks_output)):

        mask_dir = os.path.join(subfolder_masks_output, img_name)
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

        for sub_dir in os.listdir(args.image_folder):

            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:

                candidate = os.path.join(args.image_folder, sub_dir, f"{img_name}{ext}")

                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path:
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
        results = dict(sorted(results.items(), key=lambda x: int(x[0])))

        # -----------------------------
        # Save JSON
        # -----------------------------
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved {img_name} results → {json_path}")