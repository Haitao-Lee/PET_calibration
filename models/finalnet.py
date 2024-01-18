#codig = utf-8
import os
# print(os.sys.path)
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from models.nets.deeplabv3_plus import _segm_resnet
# from utils.utils import show_config

class finalnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = _segm_resnet(backbone_name="resnet18", num_classes=1, output_stride=16)
        self.fc1 = nn.Linear(65536, 512)
    
    def forward(self, x):
        out = self.model1(x)
        # print(out.shape)
        # print(out.shape)
        out = torch.reshape(out, (-1, 1, 65536))
        out = self.fc1(out)
        return out