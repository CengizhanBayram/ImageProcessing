import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(ax, image, title, color='blue'):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax.plot(hist, color=color)
    ax.set_title(title + ' Histogram')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')

# Load the original and modified images
original_image = cv2.imread(r'C:\Users\cengh\Desktop\repos\idip\D_original.png', cv2.IMREAD_GRAYSCALE)
modified_image = cv2.imread(r'C:\Users\cengh\Desktop\repos\idip\D_modified.png', cv2.IMREAD_GRAYSCALE)

# Check if the images are loaded successfully
if original_image is None or modified_image is None:
    print("Error: Could not load the image")
    exit()

# Sharpen the original image
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
sharpened_image = cv2.filter2D(original_image, -1, kernel)

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Display original image and its histogram
axs[0, 0].imshow(original_image, cmap='gray')
axs[0, 0].set_title('Original Image')
plot_histogram(axs[0, 1], original_image, 'Original', color='red')

# Display modified image and its histogram
axs[1, 0].imshow(modified_image, cmap='gray')
axs[1, 0].set_title('Modified Image')
plot_histogram(axs[1, 1], modified_image, 'Modified', color='green')

# Display sharpened image and its histogram
axs[2, 0].imshow(sharpened_image, cmap='gray')
axs[2, 0].set_title('Sharpened Image')
plot_histogram(axs[2, 1], sharpened_image, 'Sharpened', color='blue')

# Adjust spacing and show plot
plt.tight_layout()
plt.show()
