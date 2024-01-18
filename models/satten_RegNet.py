# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class _DenseLayer(nn.Sequential):
    '''DenseBlock中的内部结构, 这里是BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv结构'''
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                            kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


#实现DenseBlock模块，内部是密集连接方式（输入特征数线性增长）
class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)


class st_RegNet(nn.Module):
    def __init__(self, num_key_points=256, drop_out=0.1) -> None:
        super(st_RegNet, self).__init__()
        self.num_kp = num_key_points
        self.conv0 = nn.Conv2d(1, round(num_key_points/4), kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(round(num_key_points/4), round(num_key_points/2), kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(round(num_key_points/2), num_key_points, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(num_key_points, 2*num_key_points, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, ceil_mode=False)
        self.s_att = SpatialAttention()
        self.regress = nn.Sequential(OrderedDict([('conv3x3_0', nn.Conv2d(2*num_key_points, num_key_points, kernel_size=3, stride=1, padding=1, bias=True)),]))
        self.regress.add_module('maxpooling0', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
        self.regress.add_module('activation0', nn.ReLU(inplace=True))
        self.regress.add_module('conv3x3_1', nn.Conv2d(num_key_points, 1, 3, stride=1, padding=1, bias=True))
        self.regress.add_module('maxpooling1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
        self.regress.add_module('activation1', nn.ReLU(inplace=True))
        self.FC = nn.Linear(64*64, 512)
        self.conv_side = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        
        
    def forward(self, x):
        # out_side = self.conv_side(x) # (None, 1, 256, 256)
        # out_side = nn.Flatten(out_side, 1, 3) # (None, 1, 512)
        out1 = self.conv0(x)
        # print(out1.shape)
        out1 = self.activation(out1) #(None, 64, 256, 256)
        out2 = self.conv1(out1+x.repeat(1, round(self.num_kp/4), 1, 1)) 
        out2 = self.activation(out2) # (None, 128, 256, 256)
        # print('out2:',out2.shape)
        out3 = self.conv2(out2+out1.repeat(1, 2, 1, 1)) 
        out3 = self.activation(out3) # (None, 256, 256, 256)
        # print('out3:',out3.shape)
        out4 = self.conv3(out3+out2.repeat(1, 2, 1, 1)) # (None, 512, 256, 256)
        # print('out4:',out4.shape)
        # out = self.pooling(out4) 
        out = self.activation(out4 + out3.repeat(1,2,1,1))
        result = self.regress(out)
        result = torch.flatten(result, 2) #(None, 1, 64x64)
        result = self.FC(result)
        
        # print('out:',out.shape)
        # out = torch.chunk(out, 512, dim = 1)
        # result = self.s_att(out[0])
        # result = self.regress(result) #(None, 1, 64, 64)
        # # print('before flatten:', result.shape)
        # result = torch.flatten(result,2) #(None, 1, 64x64)
        # # print('after flatten:', result.shape)
        # result = self.FC(result) #(None, 1, 1)
        # for i in range(1, len(out)):
        #     # out_chunk = self.s_att(out[i]) #(None, 1, 256, 256)
        #     out_chunk = self.regress(out[i]) #(None, 1, 64, 64)
        #     out_chunk = torch.flatten(out_chunk,2) #(None, 1, 64x64)
        #     out_chunk = self.FC(out_chunk) #(None, 1, 1)
        #     result = torch.cat((result, out_chunk), dim=2)
        # print('result:',result.shape)
        return result
            


