import argparse
import sys
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
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




def find_mask_bbox(mask_np):
    """Find bounding box of white area in mask"""
    white_pixels = np.where(mask_np > 0)
    if len(white_pixels[0]) == 0:
        return None
    min_row, max_row = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_col, max_col = np.min(white_pixels[1]), np.max(white_pixels[1])
    return min_row, min_col, max_row, max_col




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




