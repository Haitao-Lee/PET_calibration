import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

# # 生成随机的中心点
# num_clusters = 256
# centers = np.random.randint(0, 256, size=(num_clusters, 2))

# # 创建空白的聚类结果图像
# clustered_image = np.zeros((256, 256))

# # 对每个像素进行聚类
# for i in range(256):
#     for j in range(256):
#         pixel = np.array([i, j])
#         distances = np.linalg.norm(centers - pixel, axis=1)
#         cluster_index = np.argmin(distances)
#         clustered_image[i, j] = cluster_index

# # 计算聚类边界
# def compute_cluster_boundaries(clustered_image):
#     boundaries = np.zeros_like(clustered_image, dtype=bool)
#     for i in range(1, clustered_image.shape[0]):
#         for j in range(1, clustered_image.shape[1]):
#             if clustered_image[i, j] != clustered_image[i-1, j] or clustered_image[i, j] != clustered_image[i, j-1]:
#                 boundaries[i, j] = True
#     return boundaries

# # 获取聚类边界
# boundaries = compute_cluster_boundaries(clustered_image)

# # 显示聚类结果和边界
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(clustered_image, cmap='jet')
# plt.axis('off')
# plt.title('Clustered Image')

# plt.subplot(1, 2, 2)
# plt.imshow(boundaries, cmap='gray')
# plt.axis('off')
# plt.title('Cluster Boundaries')

# plt.tight_layout()
# plt.show()


def clusterAndBoundary(img, pts):
    centers = pts
    # 创建空白的聚类结果图像
    clustered_image = np.zeros(img.shape)
    cluster_col = random.sample(range(500), 256)
    # 对每个像素进行聚类
    for i in range(256):
        for j in range(256):
            pixel = np.array([i, j])
            distances = np.linalg.norm(centers - pixel, axis=1)
            cluster_index = np.argmin(distances)
            clustered_image[i, j] = cluster_col[cluster_index]
    boundaries = np.zeros_like(clustered_image, dtype=bool)
    for i in range(1, clustered_image.shape[0]):
        for j in range(1, clustered_image.shape[1]):
            if clustered_image[i, j] != clustered_image[i-1, j] or clustered_image[i, j] != clustered_image[i, j-1]:
                boundaries[i, j] = True
    return clustered_image, boundaries