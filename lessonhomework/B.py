import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_xray(image, brightness_factor):
    """
    Enhances the given X-ray image using CLAHE and adjusts brightness.
    """
    # Convert image to grayscale if necessary
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Adjust brightness by scaling pixel values
    enhanced_lighter = cv2.addWeighted(enhanced, brightness_factor, np.zeros_like(enhanced), 0, 0)

    return enhanced_lighter

if __name__ == "__main__":
    # Load the X-ray image and the modified image
    xray_image_path = r'C:\Users\cengh\Desktop\repos\idip\B_original.png'
    modified_image_path = r'C:\Users\cengh\Desktop\repos\idip\B_modified.png'
    
    xray_image = cv2.imread(xray_image_path, cv2.IMREAD_GRAYSCALE)
    modified_image = cv2.imread(modified_image_path, cv2.IMREAD_GRAYSCALE)

    # Enhance the X-ray image and make it lighter
    brightness_factor = 1.5  # Increase brightness by a factor of 1.5 (adjust as needed)
    enhanced_xray_lighter = enhance_xray(xray_image, brightness_factor)

    # Calculate histograms
    original_hist = cv2.calcHist([xray_image], [0], None, [256], [0, 256])
    enhanced_hist = cv2.calcHist([enhanced_xray_lighter], [0], None, [256], [0, 256])
    modified_hist = cv2.calcHist([modified_image], [0], None, [256], [0, 256])

    # Plot histograms
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Display original X-ray image and its histogram
    axs[0, 0].imshow(xray_image, cmap='gray')
    axs[0, 0].set_title('Original X-ray Image')
    axs[0, 1].plot(original_hist, color='blue')
    axs[0, 1].set_title('Histogram')

    # Display enhanced X-ray image and its histogram
    axs[1, 0].imshow(enhanced_xray_lighter, cmap='gray')
    axs[1, 0].set_title('Enhanced and Lighter X-ray Image')
    axs[1, 1].plot(enhanced_hist, color='red')
    axs[1, 1].set_title('Histogram')

    # Display modified image and its histogram
    axs[2, 0].imshow(modified_image, cmap='gray')
    axs[2, 0].set_title('Modified Image')
    axs[2, 1].plot(modified_hist, color='green')
    axs[2, 1].set_title('Histogram')

    plt.tight_layout()
    plt.show()
