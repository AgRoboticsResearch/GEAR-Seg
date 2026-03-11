import argparse
import os
import torch
from utils import generate_all_sam_mask, save_description, init_models

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='/home/nya/code/code/2-Reasoning-Seg/GEAR-Seg/GEAR-Seg/img/straw', help='Path to the image folder')
    parser.add_argument('--output_folder', type=str, default='/home/nya/code/code/2-Reasoning-Seg/GEAR-Seg/GEAR-Seg/output', help='Path to save the outputs')
    parser.add_argument('--sam_model_path', type=str, default='/mnt/nas/fruit_dataset/wyn/202507/describe_anything_checkpoint/SDM-D_checkpoint/checkpoints/sam2_hiera_large.pt', help='Path to SAM model')
    parser.add_argument('--model_cfg', type=str, default="sam2_hiera_l.yaml", required = False, help='SAM2 model config file')
    parser.add_argument('--dam_model_path', type=str, 
                       default='/mnt/nas/fruit_dataset/wyn/202507/describe_anything_checkpoint/checkpoints/DAM-3B',
                       help='Path to DAM model')
    parser.add_argument('--sentence_transformer_path', type=str,
                       default='/mnt/nas/fruit_dataset/wyn/202507/describe_anything_checkpoint/checkpoints/sentence-transformers/all-MiniLM-L6-v2',
                       help='Path to Sentence Transformer model')
    # SAM parameters
    parser.add_argument('--max_image_size', type=int, default=3000, help='Max image size for SAM')
    parser.add_argument('--enable_mask_nms', type=bool, default=False, required = False,  help='Whether to apply NMS to masks')
    parser.add_argument('--mask_nms_thresh', type=float, default=0.9, required = False, help='Threshold for NMS mask overlap')
    parser.add_argument('--save_anns', type=bool, default=True, required = False,  help='Whether to save mask anns')
    parser.add_argument('--points_per_side', type=int, default=32, required = False, help='Points per side for SAM2 mask generator')
    parser.add_argument('--min_mask_region_area', type=int, default=0, required = False, help='Minimum area for a SAM2 mask to be kept')
    parser.add_argument('--crop_n_layers', type=int, default=0, required = False, help='Number of cropping layers for SAM2 mask generator')
    parser.add_argument('--stability_score_offset', type=float, default=0.7, required = False, help='The amount to shift the cutoff when calculated the stability score.')
    
    # DAM parameters
    parser.add_argument('--scale', type=float, default=1000, required = False,  help='Scale factor for expanding bounding box')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Temperature for DAM generation')
    parser.add_argument('--top_p', type=float, default=0.5,
                       help='Top p for DAM generation')
    parser.add_argument('--num_beams', type=int, default=1,
                       help='Number of beams for DAM generation')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='Max new tokens for DAM generation')
    
    
    return parser.parse_args()

def main():
    args = parse_opt()
    
    # Create output directories
    masks_folder = os.path.join(args.output_folder, 'masks')
    mask_id_visual_folder = os.path.join(args.output_folder, 'mask_id_visual')

    descriptions_folder = os.path.join(args.output_folder, 'descriptions')
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(mask_id_visual_folder, exist_ok=True)   
    os.makedirs(descriptions_folder, exist_ok=True)

    # Initialize models
    dam_model, sam_predictor= init_models(args)

    # Iterate through subfolders in the image folder
    for subfolder in os.listdir(args.image_folder):
        # subfolder = train
        subfolder_path = os.path.join(args.image_folder, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder_path}")
            
            # Define output paths for the current subfolder
            subfolder_masks_output = os.path.join(masks_folder, subfolder)
            # mask/train
            subfolder_descriptions_output = os.path.join(descriptions_folder, subfolder)
            anns_imgname_dir = os.path.join(mask_id_visual_folder, subfolder)
            os.makedirs(subfolder_masks_output, exist_ok=True)
            os.makedirs(subfolder_descriptions_output, exist_ok=True)
            os.makedirs(anns_imgname_dir, exist_ok=True)

            # # Generate all SAM masks for the images in the subfolder
            generate_all_sam_mask(
                args,
                sam_predictor, 
                subfolder_path, 
                subfolder_masks_output,
                anns_imgname_dir
            )
            
            # # Generate descriptions for all masks
            save_description(
                dam_model,
                subfolder_masks_output,
                subfolder_descriptions_output,
                args
            )

if __name__ == '__main__':
    main()


