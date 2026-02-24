import torch
import torch.nn as nn

class AdaptiveHighPassFilterModule(nn.Module):
    def __init__(self, filter_rate, corner_location=20, crop=40):
        super(AdaptiveHighPassFilterModule, self).__init__()
        self.filter_rate = filter_rate
        self.center_regions = [crop, 256 - crop]
        self.corner_regions = [
            [(corner_location, corner_location), (corner_location + crop, corner_location + crop)],
            [(corner_location, 256 - corner_location - crop), (corner_location + crop, 256 - corner_location)],
            [(256 - corner_location - crop, corner_location), (256 - corner_location, corner_location + crop)],
            [(256 - corner_location - crop, 256 - corner_location - crop), (256 - corner_location, 256 - corner_location)]
        ]

    def forward(self, x):
        B, C, H, W = x.shape
        for i in range(B):
            middle_x = x[i, :, self.center_regions[0]:self.center_regions[1], self.center_regions[0]:self.center_regions[1]]
            
            mean_values = torch.mean(middle_x, dim=(1, 2), keepdim=True)
            
            threshold = mean_values * self.filter_rate
            
            filtered_middle_x = torch.where(middle_x >= threshold, middle_x, torch.zeros_like(middle_x))
            
            x[i, :, self.center_regions[0]:self.center_regions[1], self.center_regions[0]:self.center_regions[1]] = filtered_middle_x
            
            # # Process the upper-left corner region
            # ul_x = x[i, :, self.corner_regions[0][0][0]:self.corner_regions[0][1][0], self.corner_regions[0][0][1]:self.corner_regions[0][1][1]]
            
            # # Compute the mean for the cropped region of each image in the batch
            # mean_values = torch.mean(ul_x, dim=(1, 2), keepdim=True)  # Shape: [1, 1, 1]
            
            # # Compute the threshold based on the mean values
            # threshold = mean_values * self.filter_rate  # Shape: [1, 1, 1]
            
            # # Apply the threshold to the cropped region
            # filtered_ul_x = torch.where(ul_x >= threshold, ul_x, torch.zeros_like(ul_x))  # Set values below threshold to 0
            
            # # Replace the cropped region in the original tensor with the filtered values
            # x[i, :, self.corner_regions[0][0][0]:self.corner_regions[0][1][0], self.corner_regions[0][0][1]:self.corner_regions[0][1][1]] = filtered_ul_x
            
            # # Process the upper-right corner region
            # ur_x = x[i, :, self.corner_regions[1][0][0]:self.corner_regions[1][1][0], self.corner_regions[1][0][1]:self.corner_regions[1][1][1]]

            # # Compute the mean for the cropped region of each image in the batch    
            # mean_values = torch.mean(ur_x, dim=(1, 2), keepdim=True)  # Shape: [1, 1, 1]

            # # Compute the threshold based on the mean values
            # threshold = mean_values * self.filter_rate  # Shape: [1, 1, 1]
            
            # # Apply the threshold to the cropped region
            # filtered_ur_x = torch.where(ur_x >= threshold, ur_x, torch.zeros_like(ur_x))  # Set values below threshold to 0

            # # Replace the cropped region in the original tensor with the filtered values
            # x[i, :, self.corner_regions[1][0][0]:self.corner_regions[1][1][0], self.corner_regions[1][0][1]:self.corner_regions[1][1][1]] = filtered_ur_x
            
            # # Process the lower-left corner region
            # dl_x = x[i, :, self.corner_regions[2][0][0]:self.corner_regions[2][1][0], self.corner_regions[2][0][1]:self.corner_regions[2][1][1]]

            # # Compute the mean for the cropped region of each image in the batch
            # mean_values = torch.mean(dl_x, dim=(1, 2), keepdim=True)  # Shape: [1, 1, 1]

            # # Compute the threshold based on the mean values
            # threshold = mean_values * self.filter_rate  # Shape: [1, 1, 1]
            
            # # Apply the threshold to the cropped region 
            # filtered_dl_x = torch.where(dl_x >= threshold, dl_x, torch.zeros_like(dl_x))  # Set values below threshold to 0

            # # Replace the cropped region in the original tensor with the filtered values
            # x[i, :, self.corner_regions[2][0][0]:self.corner_regions[2][1][0], self.corner_regions[2][0][1]:self.corner_regions[2][1][1]] = filtered_dl_x
            
            # # Process the lower-right corner region
            # dr_x = x[i, :, self.corner_regions[3][0][0]:self.corner_regions[3][1][0], self.corner_regions[3][0][1]:self.corner_regions[3][1][1]]

            # # Compute the mean for the cropped region of each image in the batch
            # mean_values = torch.mean(dr_x, dim=(1, 2), keepdim=True)  # Shape: [1, 1, 1]

            # # Compute the threshold based on the mean values
            # threshold = mean_values * self.filter_rate  # Shape: [1, 1, 1]
            
            # # Apply the threshold to the cropped region
            # filtered_dr_x = torch.where(dr_x >= threshold, dr_x, torch.zeros_like(dr_x))  # Set values below threshold to 0

            # # Replace the cropped region in the original tensor with the filtered values
            # x[i, :, self.corner_regions[3][0][0]:self.corner_regions[3][1][0], self.corner_regions[3][0][1]:self.corner_regions[3][1][1]] = filtered_dr_x
        
        return x