import cv2
import matplotlib.pyplot as plt

def plot_histogram(ax, image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax.plot(hist, color=color)
    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')

if __name__ == "__main__":
    # Load the original, blurred, and modified images
    original_path = r'C:\Users\cengh\Desktop\repos\idip\C_original.png'
    modified_path = r'C:\Users\cengh\Desktop\repos\idip\C_modified.png'
    
    original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(original_image, (15, 15), 0)
    modified_image = cv2.imread(modified_path, cv2.IMREAD_GRAYSCALE)

    # Plot images and histograms
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Display original image
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[1, 0].axis('off')
    plot_histogram(axs[1, 0], original_image, color='blue')

    # Display modified image
    axs[0, 1].imshow(modified_image, cmap='gray')
    axs[0, 1].set_title('Modified Image')
    axs[1, 1].axis('off')
    plot_histogram(axs[1, 1], modified_image, color='green')

    # Display blurred image
    axs[0, 2].imshow(blurred_image, cmap='gray')
    axs[0, 2].set_title('Blurred Image')
    axs[1, 2].axis('off')
    plot_histogram(axs[1, 2], blurred_image, color='red')

    plt.tight_layout()
    plt.show()
