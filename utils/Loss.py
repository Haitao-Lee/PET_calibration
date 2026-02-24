import torch
import numpy

def local_max_loss(predictions, points, radius=3, sigma=1.0):
    """
    Loss function guided by local maxima.

    Parameters:
        predictions (torch.Tensor): Model output, shape (H, W).
        points (torch.Tensor): Coordinates of target points, shape (N, 2).
        radius (int): Search radius.
        sigma (float): Standard deviation of the Gaussian kernel.
    
    Returns:
        torch.Tensor: Loss value.
    """
    H, W = predictions.shape
    device = predictions.device
    loss = 0.0

    for x, y in points:
        x, y = int(x.item()), int(y.item())
        x_min = max(x - radius, 0)
        x_max = min(x + radius + 1, H)
        y_min = max(y - radius, 0)
        y_max = min(y + radius + 1, W)

        # Extract neighborhood
        local_pred = predictions[x_min:x_max, y_min:y_max]
        local_coords = torch.stack(torch.meshgrid(
            torch.arange(x_min, x_max, device=device),
            torch.arange(y_min, y_max, device=device)
        ), dim=-1).reshape(-1, 2)

        # Compute Gaussian weights
        gaussian_weights = torch.exp(-((local_coords[:, 0] - x) ** 2 + (local_coords[:, 1] - y) ** 2) / (2 * sigma ** 2))
        gaussian_weights = gaussian_weights / gaussian_weights.sum()

        # Compute local cross-entropy loss
        local_pred = local_pred.view(-1)
        local_log_probs = torch.log(local_pred + 1e-8)
        loss -= (gaussian_weights * local_log_probs).sum()

    return loss / len(points)


def nearest_distance_loss(y_true, y_pred):
    """
    Compute the mean of the minimum Euclidean distances from each ground truth point 
    to its nearest predicted point. This loss is differentiable and can be used 
    for gradient backpropagation.
    
    Parameters:
        y_true (torch.Tensor): Ground truth coordinates, shape (B, N, D), e.g., (1, 256, 2)
        y_pred (torch.Tensor): Predicted coordinates, shape (B, N, D), e.g., (1, 256, 2)
    
    Returns:
        torch.Tensor: A scalar loss value representing the average minimum distance.
    """
    # Compute pairwise Euclidean distances between all ground truth and predicted points.
    # torch.cdist returns a distance matrix with shape (B, N, N)
    dists = torch.cdist(y_pred.reshape(-1, 256, 2),  y_true.reshape(-1, 256, 2),  p=2)
    
    # For each ground truth point, find the minimum distance to any predicted point.
    min_dists, _ = torch.min(dists, dim=2)  # shape: (B, N)
    
    # Return the mean of these minimum distances as the loss.
    loss = torch.mean(min_dists)**2
    return loss
