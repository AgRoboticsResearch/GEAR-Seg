# GEAR-Seg: Turn Pixels to Text

This project, GEAR-Seg, combines the capabilities of SAM (Segment Anything Model) and a descriptive model to process images, generate segmentation masks, and create textual descriptions for the segmented objects.

## Features

- **Image Segmentation**: Leverages the Segment Anything Model (SAM) to generate masks for objects within images.
- **Object Description**: Generates detailed textual descriptions for the content of the images.
- **Batch Processing**: Supports processing of entire folders of images.

## Project Structure
- `demo.py`: The main script to run the image processing pipeline.
- `utils.py`: Contains utility functions for mask generation and description generation, adapted from `sam2` and `describe-anything`.
- `img/`: Directory for input images.
- `output/`: Directory for output masks and descriptions.

## Usage

1. **Installation**
   ```bash
   # Clone the repository and install dependencies
   git clone <repository_url>
   cd GEAR-Seg
   pip install -r requirements.txt
   ```

2. **Run Demo**
   ```bash
   python demo.py --image_folder ./img/straw
   ```
This will process all images in the specified folder, generating masks and descriptions.
