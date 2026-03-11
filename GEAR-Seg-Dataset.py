import os
import argparse

from utils.utils import  build_output_dirs
from utils.ollama_utils import ensure_ollama_running, generate_whole_descriptions, close_ollama
from utils.reasonseg_dataset import generate_reasonseg_dataset
from utils.sam_utils import init_sam_model, generate_all_sam_mask
from utils.dam_utils import init_dam_model, save_description


# ------------------------------------------------
# parse args
# ------------------------------------------------
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--api_key", required=True)

    parser.add_argument("--output_dir", default="./outputs/reasonseg_dataset")
    
    parser.add_argument('--sam_model_path', type=str, default='./checkpoints/sam2_hiera_large.pt', help='Path to SAM model')
    parser.add_argument('--model_cfg', type=str, default="sam2_hiera_l.yaml", required = False, help='SAM2 model config file')
    
    parser.add_argument('--dam_model_path', type=str, 
                       default='./checkpoints/DAM-3B',
                       help='Path to DAM model')
    #ollama path 
    parser.add_argument("--ollama_path", default="your path to ollama", help="Path to ollama")

    # SAM parameters
    parser.add_argument('--max_image_size', type=int, default=3000, help='Max image size for SAM')
    parser.add_argument('--enable_mask_nms', type=bool, default=True, required = False,  help='Whether to apply NMS to masks')
    parser.add_argument('--mask_nms_thresh', type=float, default=0.99, required = False, help='Threshold for NMS mask overlap')
    parser.add_argument('--save_anns', type=bool, default=True, required = False,  help='Whether to save mask anns')
    parser.add_argument('--points_per_side', type=int, default=32, required = False, help='Points per side for SAM2 mask generator')
    parser.add_argument('--min_mask_region_area', type=int, default=100, required = False, help='Minimum area for a SAM2 mask to be kept')
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
    

    # API parameters
    parser.add_argument("--max_masks", type=int, default=200)
    parser.add_argument('--question_visualize', type=bool, default=False, required = False,  help='Whether to apply NMS to masks')
    

    return parser.parse_args()




# ------------------------------------------------
# Main Pipeline
# ------------------------------------------------
def main():

    args = parse_args()
    split = os.path.basename(os.path.normpath(args.image_folder))
    print("Split:", split)
  


    out = build_output_dirs(args.output_dir, split)
    args.mask_dir = out["sam_mask"]
    args.answer_json_dir = out["ans_json"]
    args.des_dir = out["des"]
    print("Output root:", os.path.join(args.output_dir, split))

    dam_model = init_dam_model(args)
    sam_predictor = init_sam_model(args)

    print("\n===== SAM segmentation =====")
    generate_all_sam_mask(args, sam_predictor, out["sam_mask"], out["sam_vis"])

    print("\n===== DAM descriptions =====")
    save_description(dam_model, out["sam_mask"], out["des"], args)

    print("\n===== Whole image description (Ollama) =====")
    ollama_process = ensure_ollama_running(args.ollama_path)
    generate_whole_descriptions(args, out["des"], model_name="llama3.2-vision:latest")

    print("\n===== Reasoning segmentation dataset generation =====")
    generate_reasonseg_dataset(args)

    close_ollama(ollama_process)


if __name__ == "__main__":
    main()