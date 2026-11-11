import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim

N = 4 # number of images to process
WRITE_IMG = False #set to True if you want to save all the processed images, False otherwise


# 1. degrading the image by applying Gaussian noise and decreasing contrast
def degrade_image(image):
    rng = np.random.default_rng(72)
    alpha, sdev = 0.8, 5                             

    x = image.astype(np.float32)
    x = 127.5 + alpha * (x - 127.5)                  
    x = x + rng.normal(0, sdev, image.shape)         
    return np.clip(np.round(x), 0, 255).astype(np.uint8)

# 2. Linear contrast stretching
def linear_contrast_stretching(image):
    # Get the minimum and maximum pixel values in the image
    min_pixel = np.min(image)
    max_pixel = np.max(image)

    # Apply linear contrast stretching formula
    stretched_image = (image - min_pixel) * (255 / (max_pixel - min_pixel))
    stretched_image = stretched_image.astype(np.uint8)
    
    return stretched_image

# 3. histogram equalization on luminance channel of cie-lab color space
def histogram_equalization(image):
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab_image)
    # equalize the histogram of the L channel, a, b unchanged
    l = cv.equalizeHist(l)
    equalized_lab_image = cv.merge((l, a, b))
    equalized_image = cv.cvtColor(equalized_lab_image, cv.COLOR_LAB2BGR)
    return equalized_image

# 4. my method: bilateral denoising with linear contrast stretching
def bilateral_denoising(image):
    denoised_image = cv.bilateralFilter(image, d=9, sigmaColor=25, sigmaSpace=75)
    return linear_contrast_stretching(denoised_image)

# clean metrics.csv file
with open("metrics.csv", "w", encoding="utf-8", newline="") as f: f.truncate(0)

# insert csv header
with open('metrics.csv', 'a') as f:
    f.write('Image,Metric,Degraded,Linear Contrast Stretching,Histogram Equalization,Bilateral Denoising\n')

for i in range(1, N+1):
    img = cv.imread(f'img/img{i}.png')
    img_degraded = degrade_image(img)
    cv.imwrite(f'img_degraded/IMG{i}_degraded.png', img_degraded)    

for i in range(1, N+1):
    img = cv.imread(f'img/img{i}.png')
    img_degraded = cv.imread(f'img_degraded/IMG{i}_degraded.png')

    img_stretched = linear_contrast_stretching(img_degraded)
    img_histogram_equalized = histogram_equalization(img_degraded)
    img_bilateral_denoised = bilateral_denoising(img_degraded)

    if WRITE_IMG:
        cv.imwrite(f'img_processed/IMG{i}_stretched.png', img_stretched)
        cv.imwrite(f'img_processed/IMG{i}_histogram_equalized.png', img_histogram_equalized)
        cv.imwrite(f'img_processed/IMG{i}_bilateral_denoised.png', img_bilateral_denoised)
    
    # write all of the images toghether
    cv.imwrite(f'img_processed/IMG{i}_all.png', np.hstack((img, img_degraded, img_stretched, img_histogram_equalized, img_bilateral_denoised)))

    # psnr calculation
    psnr_degraded = round(cv.PSNR(img, img_degraded), 2)
    psnr_stretched = round(cv.PSNR(img, img_stretched), 2)
    psnr_histogram_equalized = round(cv.PSNR(img, img_histogram_equalized), 2)
    psnr_bilateral_denoised = round(cv.PSNR(img, img_bilateral_denoised), 2)

    # ssim calculation
    ssim_degraded = round(ssim(img, img_degraded, channel_axis=2, data_range=255), 2)
    ssim_stretched = round(ssim(img, img_stretched, channel_axis=2, data_range=255), 2)
    ssim_histogram_equalized = round(ssim(img, img_histogram_equalized, channel_axis=2, data_range=255), 2)
    ssim_bilateral_denoised = round(ssim(img, img_bilateral_denoised, channel_axis=2, data_range=255), 2)

    with open('metrics.csv', 'a') as f:
        f.write(f'IMG{i},PSNR,{psnr_degraded},{psnr_stretched},{psnr_histogram_equalized},{psnr_bilateral_denoised}\n')
   
    with open('metrics.csv', 'a') as f:
        f.write(f'IMG{i},SSIM,{ssim_degraded},{ssim_stretched},{ssim_histogram_equalized},{ssim_bilateral_denoised}\n')        
