import torch
import torch.nn as nn
import torch.nn.functional as F

class GravityGuidedDebiasModule(nn.Module):
    def __init__(self, radius=1, max_iters=20, tol=1e-5, smooth_sign=True, process_sign=True, visualization_sign=False):
        """
        A PyTorch module that simulates the iterative movement of points on depth maps
        towards local maxima within a defined neighborhood radius, with support for batch processing.

        Args:
            radius (int): Radius of the local neighborhood to search for the maximum.
            max_iters (int): Maximum number of iterations for the movement process.
            tol (float): Convergence tolerance. Movement stops when maximum displacement is below this value.
        """
        super(GravityGuidedDebiasModule, self).__init__()
        self.radius = radius
        self.max_iters = max_iters
        self.tol = tol
        self.smooth_sign = smooth_sign
        self.process_sign = process_sign
        self.visulization_sign = visualization_sign


    @staticmethod
    def smooth_tensor(tensor, kernel_size=3):
        """
        Smooth the input tensor while keeping the same dimensions.
        :param tensor: Input tensor of shape (batch_size, channels, height, width).
        :param kernel_size: Size of the smoothing kernel (default: 3).
        :return: Smoothed tensor with the same dimensions as the input.
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd to ensure symmetry"
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=tensor.device) / (kernel_size ** 2)
        smoothed_tensor = F.conv2d(tensor, kernel, padding=kernel_size // 2, groups=1)
        return smoothed_tensor

    @staticmethod
    def find_adjacent_col_groups(nums):
        """
        Find groups of integers in the list where elements are adjacent (difference is 1),
        and filter out groups with fewer than 2 elements.
        :param nums: List of integers.
        :return: List of lists containing grouped adjacent integers.
        """
        if not nums:
            return []

        # Sort the numbers to ensure sequential order
        nums = sorted(nums)

        groups = []
        current_group = [nums[0]]

        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:  # Check if current number is adjacent to the previous
                current_group.append(nums[i])
            else:
                if len(current_group) > 1:  # Add group only if it has more than 1 element
                    groups.append(current_group)
                current_group = [nums[i]]  # Start a new group

        # Add the last group if it has more than 1 element
        if len(current_group) > 1:
            groups.append(current_group)

        return groups

    @staticmethod
    def find_adjacent_row_groups(nums):
        """
        Find groups of integers in the list where elements are separated by 16,
        and filter out groups with fewer than 2 elements.
        :param nums: List of integers.
        :return: List of lists containing grouped integers separated by 16.
        """
        if not nums:
            return []

        # Sort the numbers to ensure sequential order
        nums = sorted(nums)

        groups = []
        current_group = [nums[0]]

        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 16:  # Check if current number is separated by 16 from the previous
                current_group.append(nums[i])
            else:
                if len(current_group) > 1:  # Add group only if it has more than 1 element
                    groups.append(current_group)
                current_group = [nums[i]]  # Start a new group

        # Add the last group if it has more than 1 element
        if len(current_group) > 1:
            groups.append(current_group)

        return groups


    @staticmethod
    def find_overlapping_indices(points, threshold=2):
        """
        Identify groups of overlapping points based on a distance threshold for each batch.

        Args:
            points (torch.Tensor): Tensor of shape (batch_size, num_points, 2) representing the coordinates.
            threshold (float): Distance threshold below which points are considered overlapping.

        Returns:
            list: A list of lists, where each inner list contains indices of overlapping points for each batch.
        """
        batch_size, num_points, _ = points.shape
        overlapping_groups = []

        for b in range(batch_size):
            # Compute pairwise distances
            distances = torch.cdist(points[b].float().unsqueeze(0), points[b].float().unsqueeze(0), p=2).squeeze(0)  # Shape: (num_points, num_points)

            # Identify overlapping points
            visited = set()
            batch_groups = []
            for i in range(num_points):
                if i in visited:
                    continue

                # Find all points within the threshold distance
                overlapping = (distances[i] < threshold).nonzero(as_tuple=True)[0].tolist()

                # Only consider groups with more than one point
                if len(overlapping) > 1:
                    batch_groups.append(overlapping)

                # Mark all points in the group as visited
                visited.update(overlapping)

            overlapping_groups.append(batch_groups)

        return overlapping_groups

    def process_overlap_points(self, start_points, end_points):
        """
        Resolve overlapping points by retaining only the nearest point to the maximum
        and resetting others to their original positions for each batch.

        Args:
            start_points (torch.Tensor): Original positions of the points, shape (batch_size, num_points, 2).
            end_points (torch.Tensor): Updated positions of the points after movement, shape (batch_size, num_points, 2).

        Returns:
            torch.Tensor: Adjusted positions of the points, shape (batch_size, num_points, 2).
        """
        batch_size = start_points.size(0)
        
        overlap_groups = self.find_overlapping_indices(end_points)

        for b in range(batch_size):
            for overlap_group in overlap_groups[b]:
                initial_len = len(overlap_group)
                pairs = self.find_adjacent_row_groups(overlap_group)
                for pair in pairs:
                    if end_points[b, pair[0], 1] > start_points[b, pair[-1], 1]:
                        end_points[b, pair[0], :] = start_points[b, pair[0], :]
                        overlap_group.remove(pair[0])
                    elif end_points[b, pair[0], 1] < start_points[b, pair[0], 1]:
                        end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                        overlap_group.remove(pair[-1])    
                    elif pair[0] > 15:
                        diff1 = end_points[b, pair[0]] - end_points[b, pair[0] - 16]
                        diff2 = end_points[b, pair[0]] - end_points[b, pair[-1] - 16]
                        if torch.abs(diff1[1])/(torch.abs(diff1[0]) + 1e-4) < torch.abs(diff2[1])/(torch.abs(diff2[0]) + 1e-4):
                            end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                            overlap_group.remove(pair[-1])
                        else:
                            end_points[b, pair[0], :] = start_points[b, pair[0], :]
                            overlap_group.remove(pair[0])
                    elif pair[-1] < 240:
                        diff1 = end_points[b, pair[0]] - end_points[b, pair[0] + 16]
                        diff2 = end_points[b, pair[0]] - end_points[b, pair[-1] + 16]
                        if torch.abs(diff1[1])/(torch.abs(diff1[0]) + 1e-4) < torch.abs(diff2[1])/(torch.abs(diff2[0]) + 1e-4):
                            end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                            overlap_group.remove(pair[-1])
                        else:
                            end_points[b, pair[0], :] = start_points[b, pair[0], :]
                            overlap_group.remove(pair[0])                
                        
                pairs = self.find_adjacent_col_groups(overlap_group)
                for pair in pairs:
                    if end_points[b, pair[0], 0] > start_points[b, pair[-1], 0]:
                        end_points[b, pair[0], :] = start_points[b, pair[0], :]
                        overlap_group.remove(pair[0])
                    elif end_points[b, pair[0], 0] < start_points[b, pair[0], 0]:
                        end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                        overlap_group.remove(pair[-1])
                    elif pair[0] % 16 > 0:
                        diff1 = end_points[b, pair[0]] - end_points[b, pair[0] - 1]
                        diff2 = end_points[b, pair[0]] - end_points[b, pair[-1] - 1]
                        if torch.abs(diff1[0])/(torch.abs(diff1[1]) + 1e-4) < torch.abs(diff2[0])/(torch.abs(diff2[1]) + 1e-4):
                            end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                            overlap_group.remove(pair[-1])
                        else:
                            end_points[b, pair[0], :] = start_points[b, pair[0], :]
                            overlap_group.remove(pair[0])
                    elif pair[-1] % 16 < 15:
                        diff1 = end_points[b, pair[0]] - end_points[b, pair[0] + 1]
                        diff2 = end_points[b, pair[0]] - end_points[b, pair[-1] + 1]
                        if torch.abs(diff1[0])/(torch.abs(diff1[1]) + 1e-4) < torch.abs(diff2[0])/(torch.abs(diff2[1]) + 1e-4):
                            end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                            overlap_group.remove(pair[-1])
                        else:
                            end_points[b, pair[0], :] = start_points[b, pair[0], :]
                            overlap_group.remove(pair[0])
                cur_len = len(overlap_group)
                if cur_len != initial_len:
                    for idx in overlap_group:
                        end_points[b, idx, :] = start_points[b, idx, :]    
                elif len(overlap_group) > 1:
                    target_point = end_points[b, overlap_group[0]].clone()
                    min_dist = torch.inf
                    final_idx = None
                    for idx in overlap_group:
                        end_points[b, idx, :] = start_points[b, idx, :]  # Reset position
                        dist = torch.norm(start_points[b, idx] - target_point)
                        if dist < min_dist:
                            min_dist = dist
                            final_idx = idx
                    end_points[b, final_idx] = target_point
        return end_points

    def forward(self, depth_maps, start_points):
        """
        Perform iterative movement of points towards local maxima on depth maps for a batch of data.

        Args:
            depth_maps (torch.Tensor): Depth maps tensor of shape (batch_size, 1, height, width),
                                       where each pixel contains depth information.
            start_points (torch.Tensor): Tensor of initial point coordinates, shape (batch_size, num_points, 2).

        Returns:
            torch.Tensor: Final positions of points after convergence, shape (batch_size, num_points, 2).
        """
        batch_size, _, height, width = depth_maps.shape
        num_points = start_points.shape[1]
        device = depth_maps.device

        # Remove channel dimension from depth_maps
        depth_maps = depth_maps.squeeze(1)  # Shape: (batch_size, height, width)

        # Initialize positions of the points
        points = start_points.clone().float()  # Shape: (batch_size, num_points, 2)
        if self.smooth_sign:
            depth_maps = self.smooth_tensor(depth_maps)
            
        # Create relative offsets for the local neighborhood
        offsets = torch.stack(torch.meshgrid(
            torch.arange(-self.radius, self.radius + 1, device=device),
            torch.arange(-self.radius, self.radius + 1, device=device),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)  # Shape: (K, 2), where K is the number of offsets

        for _ in range(self.max_iters):
            if self.visulization_sign:
                depth_maps_tmp = torch.clone(depth_maps)
                show_points = torch.clone(points)
                show_points = show_points.reshape(-1, 256, 2)
                show_points = show_points.cpu().detach().numpy().reshape(-1, 256, 2)
                depth_maps_tmp = depth_maps_tmp.cpu().detach().numpy()
                

            # Clamp points to ensure they remain within valid map boundaries
            points[..., 0] = points[..., 0].clamp(self.radius, height - self.radius - 1)
            points[..., 1] = points[..., 1].clamp(self.radius, width - self.radius - 1)

            # Compute neighborhood coordinates for each point
            local_coords = points.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)  # Shape: (batch_size, num_points, K, 2)
            local_coords[..., 0] = local_coords[..., 0].clamp(0, height - 1)  # Clamp x-coordinates
            local_coords[..., 1] = local_coords[..., 1].clamp(0, width - 1)   # Clamp y-coordinates

            # Extract depth values for all neighborhood points
            local_x = local_coords[..., 0].long()  # Shape: (batch_size, num_points, K)
            local_y = local_coords[..., 1].long()  # Shape: (batch_size, num_points, K)

            # Make sure indices are within bounds
            local_x = torch.clamp(local_x, 0, height - 1)
            local_y = torch.clamp(local_y, 0, width - 1)

            batch_indices = torch.arange(batch_size, device=device).reshape(-1, 1, 1).expand_as(local_x)
            local_values = depth_maps[batch_indices, local_x, local_y]  # Shape: (batch_size, num_points, K)

            # Find the offset corresponding to the maximum depth value
            max_idx = torch.argmax(local_values, dim=-1)  # Shape: (batch_size, num_points)

            # Collect the coordinates of the next points
            next_points = local_coords[torch.arange(batch_size, device=device).reshape(-1, 1), 
                                       torch.arange(num_points, device=device).reshape(1, -1), 
                                       max_idx]  # Shape: (batch_size, num_points, 2)

            # Check for convergence (if the maximum displacement is below the tolerance)
            if torch.norm(next_points - points, dim=-1).max() < self.tol:
                break

            # Update positions
            points = next_points

        # Resolve overlapping points
        if self.process_sign:
            points = self.process_overlap_points(start_points, points)

        return points