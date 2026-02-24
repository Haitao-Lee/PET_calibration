from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.hub import load_state_dict_from_url
import data_visualization

# Optional imports with fallbacks
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderrmmmmmmm
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
    

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(CBR(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6), channel_attention(out_c))
        self.c3 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12), channel_attention(out_c))
        self.c4 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18), channel_attention(out_c))
        self.c5 = CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc + xs)
        x = self.sa(x)
        return x


"""Decouple Layer"""


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return f_fg, f_bg, f_uc


"""Auxiliary Head"""


class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()

        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc


class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5


        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)

        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        x = self.ca(x)
        x = self.sa(x)
        return x


class output_block(nn.Module):

    def __init__(self, in_c, out_c=1):
        super().__init__()

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.fuse=CBR(in_c*3,in_c, kernel_size=3, padding=1)
        self.c1 = CBR(in_c, 128, kernel_size=3, padding=1)
        self.c2 = CBR(128, 64, kernel_size=1, padding=0)
        self.c3 = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
        self.sig=nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x2 = self.up_2x2(x2)
        x3 = self.up_4x4(x3)

        x = torch.cat([x1, x2, x3], axis=1)
        x=self.fuse(x)

        x=self.up_2x2(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x=self.sig(x)
        return x


class multiscale_feature_aggregation(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.c11 = CBR(in_c[0], out_c, kernel_size=1, padding=0)
        self.c12 = CBR(in_c[1], out_c, kernel_size=1, padding=0)
        self.c13 = CBR(in_c[2], out_c, kernel_size=1, padding=0)
        self.c14 = CBR(out_c * 3, out_c, kernel_size=1, padding=0)

        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)

    def forward(self, x1, x2, x3):
        x1 = self.up_4x4(x1)
        x2 = self.up_2x2(x2)

        x1 = self.c11(x1)
        x2 = self.c12(x2)
        x3 = self.c13(x3)
        x = torch.cat([x1, x2, x3], axis=1)
        x = self.c14(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        return x


class CDFAPreprocess(nn.Module):

    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x



class Local_encoder(nn.Module):
    def __init__(self, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W

        """ Backbone: ResNet50 """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [batch_size, 256, h/4, w/4]
        self.layer2 = backbone.layer2  # [batch_size, 512, h/8, w/8]
        self.layer3 = backbone.layer3  # [batch_size, 1024, h/16, w/16]

        """ FEM """
        self.dconv1 = dilated_conv(64, 128)
        self.dconv2 = dilated_conv(256, 128)
        self.dconv3 = dilated_conv(512, 128)
        self.dconv4 = dilated_conv(1024, 128)

        """ Decouple Layer """
        self.decouple_layer = DecoupleLayer(1024, 128)

        """ Adjust the shape of decouple output """
        self.preprocess_fg4 = CDFAPreprocess(128, 128, 1)  # 1/16
        self.preprocess_bg4 = CDFAPreprocess(128, 128, 1)  # 1/16

        self.preprocess_fg3 = CDFAPreprocess(128, 128, 2)  # 1/8
        self.preprocess_bg3 = CDFAPreprocess(128, 128, 2)  # 1/8

        self.preprocess_fg2 = CDFAPreprocess(128, 128,4)  # 1/4
        self.preprocess_bg2 = CDFAPreprocess(128, 128, 4)  # 1/4

        self.preprocess_fg1 = CDFAPreprocess(128, 128,8)  # 1/2
        self.preprocess_bg1 = CDFAPreprocess(128, 128, 8)  # 1/2

        """ Auxiliary Head """
        self.aux_head = AuxiliaryHead(128)

        """ Contrast-Driven Feature Aggregation """
        self.up2X = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cdfa4 = ContrastDrivenFeatureAggregation(128, 128, 4)
        self.cdfa3 = ContrastDrivenFeatureAggregation(128 + 128, 128, 4)
        self.cdfa2 = ContrastDrivenFeatureAggregation(128 + 128, 128, 4)
        self.cdfa1 = ContrastDrivenFeatureAggregation(128 + 128, 128, 4)

        """ Decoder """
        self.decoder_small = decoder_block(128, 128, scale=2)
        self.decoder_middle = decoder_block(128, 128, scale=2)
        self.decoder_large = decoder_block(128, 128, scale=2)

        """ Output Block """
        self.output_block = output_block(128, 1)
        
        self.final_conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, image):
        """ Backbone: ResNet50 """
        x0 = image.repeat(1, 3, 1, 1)  ## [-1, 3, h, w]
        x1 = self.layer0(x0)  ## [-1, 64, h/2, w/2]
        x2 = self.layer1(x1)  ## [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)  ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)  ## [-1, 1024, h/16, w/16]

        """ Dilated Conv """
        d1 = self.dconv1(x1)
        d2 = self.dconv2(x2)
        d3 = self.dconv3(x3)
        d4 = self.dconv4(x4)

        """ Decouple Layer """
        f_fg, f_bg, f_uc = self.decouple_layer(x4)
        """ Auxiliary Head """
        mask_fg, mask_bg, mask_uc = self.aux_head(f_fg, f_bg, f_uc)

        """ Contrast-Driven Feature Aggregation """
        f_fg4 = self.preprocess_fg4(f_fg)
        f_bg4 = self.preprocess_bg4(f_bg)
        f_fg3 = self.preprocess_fg3(f_fg)
        f_bg3 = self.preprocess_bg3(f_bg)
        f_fg2 = self.preprocess_fg2(f_fg)
        f_bg2 = self.preprocess_bg2(f_bg)
        f_fg1 = self.preprocess_fg1(f_fg)
        f_bg1 = self.preprocess_bg1(f_bg)

        f4 = self.cdfa4(d4, f_fg4, f_bg4)
        f4_up = self.up2X(f4)
        f_4_3 = torch.cat([d3, f4_up], dim=1)
        f3 = self.cdfa3(f_4_3, f_fg3, f_bg3)
        f3_up = self.up2X(f3)
        f_3_2 = torch.cat([d2, f3_up], dim=1)
        f2 = self.cdfa2(f_3_2, f_fg2, f_bg2)
        f2_up = self.up2X(f2)
        f_2_1 = torch.cat([d1, f2_up], dim=1)
        f1 = self.cdfa1(f_2_1, f_fg1, f_bg1)


        return f1, f2, f3, f4, mask_fg, mask_bg, mask_uc



class Local_decoder(nn.Module):
    def __init__(self, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W
        
        """ Decoder """
        self.decoder_small = decoder_block(128, 128, scale=2)
        self.decoder_middle = decoder_block(128, 128, scale=2)
        self.decoder_large = decoder_block(128, 128, scale=2)

        """ Output Block """
        self.output_block = output_block(128, 1)
        
        self.final_conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, f1, f2, f3, f4, mask_fg, mask_bg, mask_uc):

        """ Decoder """
        f_small = self.decoder_small(f2, f1)
        f_middle = self.decoder_middle(f3, f2)
        f_large = self.decoder_large(f4, f3)

        """ Output Block """
        mask = self.output_block(f_small, f_middle, f_large)
        
        out = self.final_conv(torch.cat([mask, mask_fg, mask_bg, mask_uc], dim=1))

        return out






class GravityGuidedRegressionModule(nn.Module):
    def __init__(self, radius=1, max_iters=20, tol=1e-5, smooth_sign=True, process_sign=True, visualization_sign = False):
        """
        A PyTorch module that simulates the iterative movement of points on depth maps
        towards local maxima within a defined neighborhood radius, with support for batch processing.

        Args:
            radius (int): Radius of the local neighborhood to search for the maximum.
            max_iters (int): Maximum number of iterations for the movement process.
            tol (float): Convergence tolerance. Movement stops when maximum displacement is below this value.
        """
        super(GravityGuidedRegressionModule, self).__init__()
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
                    # elif pair[0] > 0 and pair[-1] < 255 and torch.norm(end_points[b, pair[0]] - start_points[b, pair[-1] + 1]) < torch.norm(end_points[b, pair[0]] - start_points[b, pair[0] - 1]):
                    #     end_points[b, pair[0], :] = start_points[b, pair[0], :]
                    #     overlap_group.remove(pair[0])
                    # elif pair[0] > 0 and pair[-1] < 255 and torch.norm(end_points[b, pair[0]] - start_points[b, pair[-1] + 1]) > torch.norm(end_points[b, pair[0]] - start_points[b, pair[0] - 1]):
                    #     end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                    #     overlap_group.remove(pair[-1])
                        
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
                    # elif pair[0] > 15 and pair[-1] < 240 and torch.norm(end_points[b, pair[0]] - start_points[b, pair[-1] + 16]) < torch.norm(end_points[b, pair[0]] - start_points[b, pair[0] - 16]):
                    #     end_points[b, pair[0], :] = start_points[b, pair[0], :]
                    #     overlap_group.remove(pair[0])
                    # elif pair[0] > 15 and pair[-1] < 240 and torch.norm(end_points[b, pair[0]] - start_points[b, pair[-1] + 16]) > torch.norm(end_points[b, pair[0]] - start_points[b, pair[0] - 16]):
                    #     end_points[b, pair[-1], :] = start_points[b, pair[-1], :]
                    #     overlap_group.remove(pair[-1])
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
                # data_visualization.image_label_visualization(depth_maps_tmp[0, :, :], show_points[0, :, 0], show_points[0, :, 1])
                

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


class MeanModelInsertModule(nn.Module):
    def __init__(self, mean_model, smooth_sign = True, process_sign = False):
        super(MeanModelInsertModule, self).__init__()
        """
        Initialize the MeanModelInsertModule.
        
        Args:
            data_set (list or dataset-like object): A dataset where each element is a tuple 
                containing the input data and its corresponding label.
        """
        super(MeanModelInsertModule, self).__init__()
        # Convert labels to a tensor
        self.mean_model = torch.tensor(mean_model).reshape(1, 256, 2)
        # Initialize the Gravity Guided Regression Module (GGRM)
        self.GGRM = GravityGuidedRegressionModule(radius= 10, smooth_sign=smooth_sign, process_sign=process_sign)

    @staticmethod
    def generate_heatmap(image_shape, points, radius=3, intensity=1.0):
        """
        Generate heatmaps for a batch of points using efficient tensor operations.
        
        Args:
            image_shape (tuple): Shape of the output image (batch_size, 1, H, W).
            points (torch.Tensor): Tensor of shape (batch_size, 1, num_points, 2), where each point is (y, x).
            radius (int): Radius of the heatmap (standard deviation of Gaussian kernel).
            intensity (float): Peak intensity of the heatmap.
        
        Returns:
            torch.Tensor: Heatmap image of shape (batch_size, 1, H, W).
        """
        _, _, height, width = image_shape
        _ = points.shape[2]

        # Generate coordinate grid for the entire image
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=points.device),
            torch.arange(width, device=points.device),
            indexing="ij"
        )
        y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

        # Expand points to match the grid
        points_y = points[..., 0].unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, num_points, 1, 1]
        points_x = points[..., 1].unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, num_points, 1, 1]

        # Compute squared distance from each pixel to each point
        dist_squared = ((y_grid - points_y) ** 2 + (x_grid - points_x) ** 2) / (radius ** 2)

        # Compute Gaussian weights
        gaussian_weights = torch.exp(-0.5 * dist_squared) * intensity  # Shape: [batch_size, 1, num_points, H, W]

        # Sum up contributions from all points
        heatmap = gaussian_weights.sum(dim=2)  # Combine all points: [batch_size, 1, H, W]

        # Clamp the heatmap to ensure valid values
        heatmap = torch.clamp(heatmap, 0, 1)
        return heatmap
    
    
    def forward(self, x):
        """
        Forward pass of the MeanModelInsertModule.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor with shape (B, 1, 512).
        """
        B, C, H, W = x.shape  # Extract batch size and dimensions from the input
        mean_m = self.mean_model.repeat(B, 1, 1).to(x.device)  # Repeat the mean model for each batch
        out = self.GGRM(x, mean_m)  # Pass input and mean model to the GGRM
        heat_map = self.generate_heatmap(x.shape, out.reshape(-1, 1, 256, 2))
        return heat_map  # Reshape output to desired shape


    
class AverageBasedFilterModule(nn.Module):
    def __init__(self, filter_rate, corner_location=20, crop=40):
        """
        Initialize the module with a given filter rate, corner location, and crop size.
        
        :param filter_rate: The rate to scale the mean for filtering.
        :param corner_location: The starting location of the corner regions.
        :param crop: The size of the crop from the image borders.
        """
        super(AverageBasedFilterModule, self).__init__()
        self.filter_rate = filter_rate
        self.center_regions = [crop, 256 - crop]
        self.corner_regions = [
            [(corner_location, corner_location), (corner_location + crop, corner_location + crop)],
            [(corner_location, 256 - corner_location - crop), (corner_location + crop, 256 - corner_location)],
            [(256 - corner_location - crop, corner_location), (256 - corner_location, corner_location + crop)],
            [(256 - corner_location - crop, 256 - corner_location - crop), (256 - corner_location, 256 - corner_location)]
        ]

    def forward(self, x):
        """
        Apply the filtering operation on the input tensor.
        
        :param x: Input tensor with shape [batch_size, 1, img_size, img_size].
        :return: Filtered tensor with the same shape as input.
        """
        B, C, H, W = x.shape
        for i in range(B):
            # Process the center region
            middle_x = x[i, :, self.center_regions[0]:self.center_regions[1], self.center_regions[0]:self.center_regions[1]]
            
            # Compute the mean for the cropped region of each image in the batch
            mean_values = torch.mean(middle_x, dim=(1, 2), keepdim=True)  # Shape: [1, 1, 1]
            
            # Compute the threshold based on the mean values
            threshold = mean_values * self.filter_rate  # Shape: [1, 1, 1]
            
            # Apply the threshold to the cropped region
            filtered_middle_x = torch.where(middle_x >= threshold, middle_x, torch.zeros_like(middle_x))  # Set values below threshold to 0
            
            # Replace the cropped region in the original tensor with the filtered values
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



class PointAdjustmentModule(nn.Module):
    """
    Module to adjust input points based on their proximity to reference points.
    This module applies a mirror transformation followed by a regression adjustment 
    (via the Gravity Guided Regression Module, GGRM) to refine points that are within 
    a specified distance threshold.
    """
    def __init__(self, threshold, lamb=1, smooth_sign=True, process_sign=False):
        """
        Initialize the point adjustment module.

        Args:
            threshold (float): Distance threshold to determine if a point should be adjusted.
            smooth_sign (bool): Whether to use smooth sign in GGRM (default: True).
            process_sign (bool): Flag for additional processing in GGRM (default: False).
        """
        super(PointAdjustmentModule, self).__init__()
        self.threshold = threshold
        self.GGRM = GravityGuidedRegressionModule(smooth_sign=smooth_sign, process_sign=process_sign)
        self.lamb = lamb

    def forward(self, x, tensor1, tensor2):
        """
        Adjust the points in tensor1 based on their proximity to reference points in tensor2.

        Args:
            x: Additional input required by GGRM (the meaning depends on its implementation).
            tensor1 (torch.Tensor): Points to be adjusted, shape [batch_size, 1, 256, 2].
            tensor2 (torch.Tensor): Reference points, shape [batch_size, 1, 256, 2].

        Returns:
            torch.Tensor: Adjusted tensor1 with the same shape as the input.
        """
        assert tensor1.shape == tensor2.shape, f"Input tensor shapes mismatch: {tensor1.shape} vs {tensor2.shape}"

        # Check for NaN or Inf values
        if torch.isnan(tensor1).any() or torch.isnan(tensor2).any():
            raise ValueError("Input tensors contain NaN values.")
        if torch.isinf(tensor1).any() or torch.isinf(tensor2).any():
            raise ValueError("Input tensors contain Inf values.")

        # First mirror transformation
        # inverse_tensor = torch.round(tensor2 + self.lamb * (tensor1 - tensor2))
        # valid_tensor = self.GGRM(x, inverse_tensor.reshape(-1, 256, 2))
        # valid_tensor = valid_tensor.reshape_as(tensor1)
        # distances = torch.norm(valid_tensor - tensor2, dim=-1)
        # indices1 = torch.where(distances < self.threshold)

        # Second mirror transformation
        inverse_tensor = torch.round(tensor2 - (self.lamb) * (tensor1 - tensor2))
        valid_tensor = self.GGRM(x, inverse_tensor.reshape(-1, 256, 2))
        valid_tensor = valid_tensor.reshape_as(tensor1)
        distances = torch.norm(valid_tensor - tensor2, dim=-1)
        indices1 = torch.where(distances < self.threshold)

        # Get intersection of valid indices based on last dimension (point index)
        common_idx = torch.isin(indices1[2], indices1[2])
        filtered_i0 = indices1[0][common_idx]
        filtered_i1 = indices1[1][common_idx]
        filtered_i2 = indices1[2][common_idx]

        # # Additional filtering: only keep indices where mod/div of 16 ∈ [5,10]
        # mod_16 = filtered_i2 % 16
        # div_16 = filtered_i2 // 16
        # corner_mask = ((div_16 < 5) & (mod_16 < 5)) | ((div_16 < 5) & (mod_16 > 10)) | ((div_16 > 10) & (mod_16 < 5)) | ((div_16 > 10) & (mod_16 > 10)) 
        # # Final indices after all filtering
        # update_idx = (filtered_i0[~corner_mask], filtered_i1[~corner_mask], filtered_i2[~corner_mask])
        update_idx = (filtered_i0, filtered_i1, filtered_i2)

        # Apply update
        tensor1[update_idx[0], update_idx[1], update_idx[2], :] = \
            valid_tensor[update_idx[0], update_idx[1], update_idx[2], :]

        return tensor1
        


class DynamicFusionModule(nn.Module):
    def __init__(self, threshold=5.0, sharpness=2):
        super(DynamicFusionModule, self).__init__()
        self.threshold = threshold  # Threshold to determine the transition point
        self.sharpness = sharpness  # Sharpness controls the steepness of the transition

    def forward(self, x1, x2):
        # Compute the element-wise absolute difference
        diff = torch.abs(x1 - x2)  # Shape: (batch_size, 1, 512)
        
        # Calculate weights based on the difference
        fusion_weights = torch.sigmoid(self.sharpness * (diff - self.threshold))  # Shape: (batch_size, 1, 512)
        
        # Weight x1 and x2 accordingly
        output = (1 - fusion_weights) * x1 + fusion_weights * x2  # Shape: (batch_size, 1, 512)

        return output


class DownsampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3,  # Kernel size
            stride=2,       # Stride, reduces size by half
            padding=1       # Padding to maintain center alignment
        )
        self.norm = nn.BatchNorm2d(out_channels)  # Optional: normalization
        self.activation = nn.ReLU()              # Optional: activation function

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)



class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out



class Compressor(nn.Module):
    def __init__(self, channels, reduction_factor=8):
        super(Compressor, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.model(x)
    



# def compute_filtered_tensor(tensor_A: torch.Tensor, tensor_B: torch.Tensor, mse_threshold: float) -> torch.Tensor:
#     """
#     Resize two tensors from shape [-1,1,512] to [-1,256,2],
#     compute the mean squared error (MSE) for each corresponding point,
#     and return a new tensor where for each point, if the MSE is greater than
#     the given threshold, the point's value is taken from tensor_A; otherwise, from tensor_B.

#     Parameters:
#         tensor_A (torch.Tensor): Tensor of shape [-1,1,512].
#         tensor_B (torch.Tensor): Tensor of shape [-1,1,512].
#         mse_threshold (float): Threshold for the mean squared error.

#     Returns:
#         torch.Tensor: Resulting tensor of shape [1,256,2] after applying the filtering.
#     """
#     # Reshape the tensors to [1,256,2]
#     A_reshaped = tensor_A.reshape(-1, 256, 2)
#     B_reshaped = tensor_B.reshape(-1, 256, 2)
    
#     # Compute the MSE for each corresponding point (over the last dimension)
#     mse_per_point = torch.mean((A_reshaped - B_reshaped) ** 2, dim=2)  # Shape: [1,256]
    
#     # Create a mask where mse > mse_threshold.
#     # Expand dimensions to [1,256,1] for broadcasting with the last dimension.
#     mask = (mse_per_point > mse_threshold).unsqueeze(2)
    
#     # For each point, if mask is True, take C = A, else take C = B.
#     result = torch.where(mask, A_reshaped, B_reshaped)
    
#     return result



class LocalizationNetwork(nn.Module):
    def __init__(self, 
                 mean_model, 
                 filter_rate = 1, 
                 img_size = (256, 256),
                 feat_size=[64, 128, 256, 512],
                 point_distance_threshold=4,
                 error_threshold = 20**2,
                 ABFM_sign = True,
                 MMIM_sign = True,
                 GGRM_sign = True,      
                 smooth_sign = True,
                 process_sign = True,
                 ):
        super(LocalizationNetwork, self).__init__()
        self.ABFM_sign = ABFM_sign
        self.MMIM_sign = MMIM_sign
        self.GGRM_sign = GGRM_sign
        self.error_threshold = error_threshold
        self.mean_model = torch.tensor(mean_model).reshape(-1, 1, 512)
        self.MMIM = MeanModelInsertModule(mean_model)
        self.GGRM = GravityGuidedRegressionModule(smooth_sign=smooth_sign, process_sign=process_sign, visualization_sign=False)
        self.ABFM = AverageBasedFilterModule(filter_rate)
        self.PAM = PointAdjustmentModule(point_distance_threshold)
        
        self.encoder = Local_encoder(img_size[0], img_size[1])          
 
        self.decoder = Local_decoder(img_size[0], img_size[1])  
        self.regression_head = nn.Linear(256, 2)
    @staticmethod
    def normalize_01(x):
        """
        Normalize data to the range [0, 1].

        Parameters:
        - x (array-like): Input data.

        Returns:
        - array-like: Normalized data.
        """
        _min = x.min()
        _max = x.max()
        if _max != _min:
            return (x - _min) / (_max - _min)
        else:
            return x / _min if _min != 0 else x  # Avoid division by zero
    def forward(self, x):
        if self.ABFM_sign:
            x = self.ABFM(x)
        if self.MMIM_sign:
            x = self.MMIM(x)*x + x
        x = self.normalize_01(x)
        f1, f2, f3, f4, mask_fg, mask_bg, mask_uc = self.encoder(x)
        out = self.decoder(f1, f2, f3, f4, mask_fg, mask_bg, mask_uc)
        out = self.regression_head(out).reshape(-1, 1, 512)
        # out = compute_filtered_tensor(self.mean_model.repeat(out.shape[0], 1, 1).to(device=out.device), out, self.error_threshold)
        if self.GGRM_sign and not torch.is_grad_enabled():
            out = self.PAM(x, out.reshape(-1, 1, 256, 2).clone(), self.GGRM(x, out.reshape(-1, 256, 2).clone()).reshape(-1, 1, 256, 2))
            out = torch.clamp(out, min=0, max=255)
        return out.reshape(-1, 1, 512)
        
        
def LocalNet(mean_model_path):
    mean_model = torch.load(mean_model_path)
    model = LocalizationNetwork(mean_model)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    return model