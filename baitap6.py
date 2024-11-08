import cv2
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import matplotlib.pyplot as plt



# Đường dẫn tới hai ảnh vệ tinh
image_paths = ['anh2.png', 'anh1.jpg']
def process_and_cluster_image(image_path, n_clusters=3):
    # 1. Đọc ảnh và chuyển về RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Chuyển ảnh thành mảng 2D cho phân cụm
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 2. Phân cụm KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_labels = kmeans.fit_predict(pixel_values)
    kmeans_result = kmeans_labels.reshape(image.shape[:2])

    # 3. Phân cụm Fuzzy C-Means (FCM)
    pixel_values_transpose = pixel_values.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        pixel_values_transpose, n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )
    fcm_labels = np.argmax(u, axis=0)
    fcm_result = fcm_labels.reshape(image.shape[:2])

    return image, kmeans_result, fcm_result
# 4. Hiển thị kết quả
plt.figure(figsize=(15, 10))
for i, image_path in enumerate(image_paths):
    image, kmeans_result, fcm_result = process_and_cluster_image(image_path)

    plt.subplot(2, 3, 3 * i + 1)
    plt.title(f'Ảnh gốc {i+1}')
    plt.imshow(image)

    plt.subplot(2, 3, 3 * i + 2)
    plt.title(f'KMeans Clustering {i+1}')
    plt.imshow(kmeans_result, cmap='viridis')

    plt.subplot(2, 3, 3 * i + 3)
    plt.title(f'FCM Clustering {i+1}')
    plt.imshow(fcm_result, cmap='viridis')

plt.tight_layout()
plt.show()
