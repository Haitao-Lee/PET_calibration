import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from sklearn.cluster import KMeans, DBSCAN


def distance_transform(binary_img):
    binary_img = (binary_img == 0)
    return distance_transform_edt(binary_img)


def fine_tune_position_by_maximization(original_img, clustered_img, peaks):
    for i in range(peaks.shape[0]):
        cluster_label = clustered_img[peaks[i][0], peaks[i][1]]
        cluster_indices = np.argwhere(clustered_img == cluster_label)
        cluster_values = original_img[cluster_indices[:, 0], cluster_indices[:, 1]]
        max_index = np.argmax(cluster_values)
        peaks[i, :] = cluster_indices[max_index]
    return peaks


def fine_tune_position_by_average_center(original_img, clustered_img, peaks):
    for i in range(peaks.shape[0]):
        cluster_label = clustered_img[peaks[i][0], peaks[i][1]]
        cluster_indices = np.argwhere(clustered_img == cluster_label)
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_values = original_img[cluster_indices[:, 0], cluster_indices[:, 1]]
        average_value = np.mean(cluster_values)
        
        mask = np.zeros_like(original_img, dtype=bool)
        mask[cluster_indices[:, 0], cluster_indices[:, 1]] = cluster_values > average_value
        
        distance_map = distance_transform_edt(mask)
        max_idx = np.unravel_index(np.argmax(distance_map), distance_map.shape)
        peaks[i, :] = max_idx
        
    return peaks


def largest_connected_island(coords, connectivity=4):
    if len(coords) == 0:
        return np.array([])

    coord_set = set(map(tuple, coords))
    visited = set()
    largest_island = []

    if connectivity == 4:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    elif connectivity == 8:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        raise ValueError("Connectivity must be 4 or 8")

    def dfs(coord):
        stack = [coord]
        island = []
        while stack:
            current = stack.pop()
            if current not in visited and current in coord_set:
                visited.add(current)
                island.append(current)
                for dx, dy in directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor in coord_set and neighbor not in visited:
                        stack.append(neighbor)
        return island

    for coord in coord_set:
        if coord not in visited:
            island = dfs(coord)
            if len(island) > len(largest_island):
                largest_island = island

    return np.array(largest_island)


def generate_cluster_col(n_clusters):
    return np.array(random.sample(range(256), n_clusters))


def get_boundaries(label_img):
    boundaries = np.zeros_like(label_img, dtype=bool)
    boundaries[1:, :] |= (label_img[1:, :] != label_img[:-1, :])
    boundaries[:, 1:] |= (label_img[:, 1:] != label_img[:, :-1])
    return boundaries


def clusterAndBoundary_kmeans(img, pts):
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels = np.stack((yy.ravel(), xx.ravel()), axis=1)

    kmeans = KMeans(n_clusters=pts.shape[0], init=pts, n_init=1, max_iter=1)
    cluster_indices = kmeans.fit_predict(pixels)

    clustered_image = cluster_indices.reshape(H, W)
    boundaries = get_boundaries(clustered_image)

    return clustered_image, boundaries


def method_kdtree(img, pts):
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels = np.stack((yy.ravel(), xx.ravel()), axis=-1)
    
    tree = KDTree(pts)
    _, cluster_indices = tree.query(pixels)
    
    cluster_col = generate_cluster_col(pts.shape[0])
    label_img = cluster_col[cluster_indices].reshape(H, W)
    boundaries = get_boundaries(label_img)
    
    return label_img, boundaries


def method_distance_transform(img, pts):
    H, W = img.shape
    mask = np.zeros((H, W, pts.shape[0]), dtype=bool)
    for i, (y, x) in enumerate(pts):
        if 0 <= y < H and 0 <= x < W:
            mask[int(y), int(x), i] = True

    dist_map = np.stack([distance_transform_edt(~m) for m in np.rollaxis(mask, 2)], axis=-1)
    cluster_indices = np.argmin(dist_map, axis=-1)
    
    cluster_col = generate_cluster_col(pts.shape[0])
    label_img = cluster_col[cluster_indices]
    boundaries = get_boundaries(label_img)
    
    return label_img, boundaries


def method_watershed(img, pts):
    H, W = img.shape
    markers = np.zeros((H, W), dtype=int)
    for i, (y, x) in enumerate(pts):
        if 0 <= y < H and 0 <= x < W:
            markers[int(y), int(x)] = i + 1

    elevation_map = np.ones_like(img)
    label_img = watershed(elevation_map, markers)
    boundaries = get_boundaries(label_img)
    
    return label_img, boundaries


def method_potential_field(img, pts, sigma=20.0):
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pot = np.zeros((H, W, pts.shape[0]))

    for i, (y, x) in enumerate(pts):
        pot[:, :, i] = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * sigma**2))

    cluster_indices = np.argmax(pot, axis=-1)
    cluster_col = generate_cluster_col(pts.shape[0])
    label_img = cluster_col[cluster_indices]
    boundaries = get_boundaries(label_img)
    
    return label_img, boundaries


def clusterAndBoundary(img, pts):
    H, W = img.shape
    n_centers = pts.shape[0]

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels = np.stack((yy.ravel(), xx.ravel()), axis=-1)

    dists = np.linalg.norm(pixels[:, None, :] - pts[None, :, :], axis=2)
    cluster_indices = np.argmin(dists, axis=1)

    cluster_col = generate_cluster_col(n_centers)
    clustered_image = cluster_col[cluster_indices].reshape(H, W)

    boundaries = get_boundaries(clustered_image)

    return clustered_image, boundaries