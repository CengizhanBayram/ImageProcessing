import cv2
import numpy as np
from matplotlib import pyplot as plt

# File paths
original_file_path = r"C:\Users\cengh\Desktop\repos\idip\F_original.png"
modified_file_path = r"C:\Users\cengh\Desktop\repos\idip\F_modified.png"

# Read the images
original_image = cv2.imread(original_file_path)
modified_image = cv2.imread(modified_file_path)

# Convert the images to grayscale
original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
modified_gray = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)

# Normalize the original image
original_normalized = original_gray / 255.0

# Apply gamma correction to the original image
gamma = 0.6  # You can adjust the gamma value
gamma_corrected = np.power(original_normalized, gamma)

# Convert the gamma-corrected image back to the range 0-255
gamma_corrected = np.uint8(gamma_corrected * 255)

# Calculate histograms of the original, gamma-corrected, and modified images
hist_original = cv2.calcHist([original_gray], [0], None, [256], [0, 256])
hist_gamma_corrected = cv2.calcHist([gamma_corrected], [0], None, [256], [0, 256])
hist_modified = cv2.calcHist([modified_gray], [0], None, [256], [0, 256])

# Show the result
plt.figure(figsize=(12, 8))

# Display original image and its histogram
plt.subplot(2, 3, 1)
plt.imshow(original_gray, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 4)
plt.plot(hist_original, color='blue')
plt.title('Original Image Histogram')

# Display gamma-corrected image and its histogram
plt.subplot(2, 3, 2)
plt.imshow(gamma_corrected, cmap='gray')
plt.title('Gamma Corrected Image')

plt.subplot(2, 3, 5)
plt.plot(hist_gamma_corrected, color='red')
plt.title('Gamma Corrected Image Histogram')

# Display modified image and its histogram
plt.subplot(2, 3, 3)
plt.imshow(modified_gray, cmap='gray')
plt.title('Modified Image')

plt.subplot(2, 3, 6)
plt.plot(hist_modified, color='green')
plt.title('Modified Image Histogram')

plt.tight_layout()
plt.show()
