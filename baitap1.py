import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
image = cv2.imread('anh4.jpg', cv2.IMREAD_GRAYSCALE)

# Negative Image
negative_image = 255 - image

# Increase Contrast using Contrast Stretching
def contrast_stretching(img):
    a = np.min(img)
    b = np.max(img)
    stretched = 255 * (img - a) / (b - a)
    return np.uint8(stretched)

contrast_image = contrast_stretching(image)

# Log Transformation
def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    return np.uint8(log_image)

log_image = log_transform(image)

# Histogram Equalization
hist_equalized_image = cv2.equalizeHist(image)

# Display results
titles = ['Original Image', 'Negative Image', 'Contrast Image', 'Log Image', 'Histogram Equalization']
images = [image, negative_image, contrast_image, log_image, hist_equalized_image]

plt.figure(figsize=(12, 8))

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

# Save the results
cv2.imwrite('negative_image.jpg', negative_image)
cv2.imwrite('contrast_image.jpg', contrast_image)
cv2.imwrite('log_image.jpg', log_image)
cv2.imwrite('hist_equalized_image.jpg', hist_equalized_image)