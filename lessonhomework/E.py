import cv2
import numpy as np
from matplotlib import pyplot as plt

# Dosya yolu
original_file_path = r"C:\Users\cengh\Desktop\repos\idip\E_original.png"
modified_file_path = r"C:\Users\cengh\Desktop\repos\idip\E_modified.png"

# Fotoğrafları oku
original_image = cv2.imread(original_file_path)
modified_image = cv2.imread(modified_file_path)

# Median filtresi uygula
median_filtered_image = cv2.medianBlur(original_image, 11)  # 11x11 kernel boyutu

# Orjinal görüntü histogramını hesapla
hist_original = cv2.calcHist([original_image], [0], None, [256], [0, 256])

# Median filtresi uygulanmış görüntü histogramını hesapla
hist_median_filtered = cv2.calcHist([median_filtered_image], [0], None, [256], [0, 256])

# Modifiye edilmiş görüntü histogramını hesapla
hist_modified = cv2.calcHist([modified_image], [0], None, [256], [0, 256])

# Orijinal ve median filtresi uygulanmış görüntüleri göster
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Median Filtered Image')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
plt.title('Modified Image')

# Orijinal görüntü histogramını göster
plt.subplot(2, 3, 4)
plt.plot(hist_original, color='blue')
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Original Image Histogram')

# Median filtresi uygulanmış görüntü histogramını göster
plt.subplot(2, 3, 5)
plt.plot(hist_median_filtered, color='red')
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Median Filtered Image Histogram')

# Modifiye edilmiş görüntü histogramını göster
plt.subplot(2, 3, 6)
plt.plot(hist_modified, color='green')
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.title('Modified Image Histogram')

plt.tight_layout()
plt.show()
import cv2
print("OpenCV sürümü:", plt.__version__)
