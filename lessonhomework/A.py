import cv2
import matplotlib.pyplot as plt

# Load the original and modified images
original_image = cv2.imread(r'C:\Users\cengh\Desktop\repos\idip\A_original.png', cv2.IMREAD_GRAYSCALE)
modified_image = cv2.imread(r'C:\Users\cengh\Desktop\repos\idip\A_modified.png', cv2.IMREAD_GRAYSCALE)

# Check if the images are loaded successfully
if original_image is None or modified_image is None:
    print("Error: Could not load the image")
    exit()

# Calculate the negative image
negative_image = 255 - original_image

# Calculate histograms
original_hist = cv2.calcHist([original_image], [0], None, [256], [0, 256])
modified_hist = cv2.calcHist([modified_image], [0], None, [256], [0, 256])
negative_hist = cv2.calcHist([negative_image], [0], None, [256], [0, 256])

# Plot histograms
plt.figure(figsize=(15, 10))

# Display original image
plt.subplot(2, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

# Display modified image
plt.subplot(2, 3, 2)
plt.imshow(modified_image, cmap='gray')
plt.title('Modified Image')

# Orijinal görüntü histogramını göster
plt.subplot(2, 3, 4)
plt.plot(original_hist, color='blue')
plt.title('Original Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Median filtresi uygulanmış görüntü histogramını göster
plt.subplot(2, 3, 5)
plt.plot(modified_hist, color='green')
plt.title('Modified Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Display negative image
plt.subplot(2, 3, 3)
plt.imshow(negative_image, cmap='gray')
plt.title('Negative Image')

# Negative image histogram
plt.subplot(2, 3, 6)
plt.plot(negative_hist, color='red')
plt.title('Negative Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
