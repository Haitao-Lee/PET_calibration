import math
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


class denseNet121_reg(nn.Module):
    def __init__(self, num_key_points=256, drop_out=0.1) -> None:
        super(denseNet121_reg, self).__init__()
        self.num_kp = num_key_points
        self.backbone = models.densenet121()
        self.FC = nn.Linear(1000, 512)
        
    def forward(self, x):
        out = self.backbone(torch.cat([x,x,x], dim=1))
        out = self.FC(out)
        out = torch.unsqueeze(out,dim=1)
        # print(out.shape)
        return out

class denseNet169_reg(nn.Module):
    def __init__(self, num_key_points=256, drop_out=0.1) -> None:
        super(denseNet169_reg, self).__init__()
        self.num_kp = num_key_points
        self.backbone = models.densenet169()
        self.FC = nn.Linear(1000, 512)
        
    def forward(self, x):
        out = self.backbone(torch.cat([x,x,x], dim=1))
        out = self.FC(out)
        out = torch.unsqueeze(out,dim=1)
        # print(out.shape)
        return out
