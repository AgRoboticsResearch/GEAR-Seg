import os
import numpy as np
import torch
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator



def filter_masks_by_overlap(masks, threshold):

    masks_np = [np.array(mask['segmentation'], dtype=np.bool_) for mask in masks]

    areas = [mask.sum() for mask in masks_np]

    scores = [mask['stability_score'] for mask in masks]

    keep = torch.ones(len(masks_np), dtype=torch.bool)

    for i in range(len(masks_np)):

        if not keep[i]:
            continue

        for j in range(i + 1, len(masks_np)):

            if not keep[j]:
                continue

            intersection = np.logical_and(masks_np[i], masks_np[j]).sum()

            if intersection == 0:
                continue

            overlap_i = intersection / areas[i]
            overlap_j = intersection / areas[j]

            # 只有两个比例都超过阈值才认为是重复
            if overlap_i > threshold and overlap_j > threshold:

                if scores[i] < scores[j]:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    filtered_masks = [mask for idx, mask in enumerate(masks) if keep[idx]]

    return filtered_masks


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
        y, x = np.mean(np.argwhere(m), axis=0).astype(int)
        ax.text(x, y, str(i), color='white', fontsize=15, ha='center', va='center', weight='bold')
    ax.imshow(img)
    plt.axis('off')
    save_path = os.path.join(anns_imgname_dir, f'{img_name}.png')
    plt.savefig(save_path)
    plt.close(fig)

def save_mask(anns, imgname_dir):
    os.makedirs(imgname_dir, exist_ok=True)
    for i, ann in enumerate(anns):
        mask = ann['segmentation']
        mask = np.stack([mask]*3, axis=-1)
        img = (mask*255).astype(np.uint8)
        cv2.imwrite(f'{imgname_dir}/mask_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def init_sam_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # if device.type == "cuda":
    #     # use bfloat16 for the entire notebook
    #     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    #     if torch.cuda.get_device_properties(0).major >= 8:
    #         torch.backends.cuda.matmul.allow_tf32 = True
    #         torch.backends.cudnn.allow_tf32 = True
    # elif device.type == "mps":
    #     print(
    #         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
    #         "give numerically different outputs and sometimes degraded performance on MPS. "
    #         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    #     )  
    # torch.cuda.empty_cache()
    sam_model = build_sam2(args.model_cfg, args.sam_model_path, device=device)
    return SAM2AutomaticMaskGenerator(sam_model, 
                                      points_per_side=args.points_per_side,
                                      min_mask_region_area=args.min_mask_region_area, 
                                      crop_n_layers=args.crop_n_layers, 
                                      stability_score_offset=args.stability_score_offset)


def generate_all_sam_mask(args, sam_predictor, masks_output_folder, anns_imgname_dir):
    """
    Generates and saves SAM masks for all images in a folder.
    """
    image_files = [f for f in os.listdir(args.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(args.image_folder, image_file)
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