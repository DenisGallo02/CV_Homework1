# CV_Homework1

## Requirements and Dependencies
- **Python Version:** Python 3.12.3
- **Dependencies:** 
  - `numpy`
  - `opencv-python` (cv2)
  - `scikit-image` (required for SSIM metric calculation)

You can install the necessary dependencies using:
`pip install numpy opencv-python scikit-image`

## How to Run the Experiment
To execute the complete pipeline from the input images and produce the processed results and metrics, run the following command in your terminal:

`python3 hw1.py`

## File Structure
```text
.
├── hw1.py                  # Main Python script containing the pipeline
├── README.md               # This documentation file
├── metrics.csv             # Output table containing PSNR and SSIM metrics
├── img/                    # Directory containing original reference images
├── img_degraded/           # Directory where synthetically degraded images are saved
└── img_processed/          # Directory containing enhanced results and side-by-side comparisons


You can set the flag WRITE_IMG to True at the beginning of the code if you want to put in img_processed all the single images of the pipeline. False if you want only the side by side

Set N as the number of images in img folder. 4 is the default with my images.