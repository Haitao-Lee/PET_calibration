import torch
import torch.nn as nn
from GGDM import GravityGuidedDebiasModule as GravityGuidedDebiasModule

class MeanModelFusionModule(nn.Module):
    def __init__(self, mean_model, smooth_sign=True, process_sign=False):
        super(MeanModelFusionModule, self).__init__()
        self.mean_model = torch.tensor(mean_model).reshape(1, 256, 2)
        self.GGDM = GravityGuidedDebiasModule(radius=10, smooth_sign=smooth_sign, process_sign=process_sign)

    @staticmethod
    def generate_heatmap(image_shape, points, radius=3, intensity=1.0):
        _, _, height, width = image_shape

        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=points.device),
            torch.arange(width, device=points.device),
            indexing="ij"
        )
        y_grid = y_grid.unsqueeze(0).unsqueeze(0)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)

        points_y = points[..., 0].unsqueeze(-1).unsqueeze(-1)
        points_x = points[..., 1].unsqueeze(-1).unsqueeze(-1)

        dist_squared = ((y_grid - points_y) ** 2 + (x_grid - points_x) ** 2) / (radius ** 2)

        gaussian_weights = torch.exp(-0.5 * dist_squared) * intensity

        heatmap = gaussian_weights.sum(dim=2)

        heatmap = torch.clamp(heatmap, 0, 1)
        return heatmap
    
    def forward(self, x):
        B, _, _, _ = x.shape
        mean_m = self.mean_model.repeat(B, 1, 1).to(x.device)
        out = self.GGDM(x, mean_m)
        heat_map = self.generate_heatmap(x.shape, out.reshape(-1, 1, 256, 2))
        return heat_map