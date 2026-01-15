# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules - MAFPN with RepHMS_LALK integrated."""
 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import repeat
import collections.abc

 
# ==================== 基础模块（只保留一个Conv定义）====================

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class AVG(nn.Module):
    def __init__(self, down_n=2):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.down_n = down_n
 
    def forward(self, x):
        B, C, H, W = x.shape
        H = int(H / self.down_n)
        W = int(W / self.down_n)
        output_size = np.array([H, W])
        x = self.avg_pool(x, output_size)
        return x


# ==================== 辅助函数 ====================

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

def get_conv2d_uni(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), dtype=kernel.dtype, device=kernel.device)
    if kernel.size(1) == 1:
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)
 
def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel

def get_bn(channels):
    return nn.BatchNorm2d(channels)


# ==================== UniRepLKNet 核心模块 ====================

class DilatedReparamBlock(nn.Module):
    """Dilated Reparam Block proposed in UniRepLKNet"""
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d_uni(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy)
        self.attempt_use_lk_impl = attempt_use_lk_impl
 
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [7, 5, 3]
            self.dilates = [1, 1, 1]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3]
            self.dilates = [1, 1]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        elif kernel_size == 3:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')
 
        if not deploy:
            self.origin_bn = get_bn(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels))
 
    def forward(self, x):
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out
 
    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d_uni(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class UniRepLKNetBlock(nn.Module):
    """UniRepLKNet Block - 静态可重参数化大核"""
    def __init__(self, dim, kernel_size, deploy=False, attempt_use_lk_impl=True):
        super().__init__()
        if deploy:
            print('------------------------------- Note: deploy mode')
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 3:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            assert kernel_size in [3]
            self.dwconv = get_conv2d_uni(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=deploy,
                                     attempt_use_lk_impl=attempt_use_lk_impl)
 
        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim)
 
    def forward(self, inputs):
        out = self.norm(self.dwconv(inputs))
        return out
 
    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                            self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 self.dwconv.padding, self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()


# ==================== 🔥 新增：LALK注意力模块 ====================

class CoordinateAttention(nn.Module):
    """Coordinate Attention (CVPR 2021) - 对小目标特别有效"""
    def __init__(self, channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_w * a_h
        return out


class SpatialGate(nn.Module):
    """轻量级空间门控（比CA更轻量）"""
    def __init__(self, channels):
        super(SpatialGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.gate(x)


# ==================== 原始RepHMS模块 ====================

class DepthBottleneckUni(nn.Module):
    """原始DepthBottleneckUni"""
    def __init__(self, in_channels, out_channels, shortcut=True, kersize=5,
                 expansion_depth=1, small_kersize=3, use_depthwise=True):
        super(DepthBottleneckUni, self).__init__()
        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, out_channels, k=1)  # 🔥 修正：k而不是kernel_size
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)
 
    def forward(self, x):
        y = self.conv1(x)
        y = self.act(self.conv2(y))
        y = self.one_conv(y)
        return y


class RepHDW(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut=True, expansion=0.5, 
                 kersize=5, depth_expansion=1, small_kersize=3, use_depthwise=True):
        super(RepHDW, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, 
                                                    depth_expansion, small_kersize, use_depthwise) 
                               for _ in range(depth))
        self.conv2 = Conv(c_ * (depth+2), out_channels, 1, 1)
 
    def forward(self, x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return y_out


class DepthBottleneckUniv2(nn.Module):
    """原始DepthBottleneckUniv2"""
    def __init__(self, in_channels, out_channels, shortcut=True, kersize=5,
                 expansion_depth=1, small_kersize=3, use_depthwise=True):
        super(DepthBottleneckUniv2, self).__init__()
        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, k=1)  # 🔥 修正
            self.conv3 = UniRepLKNetBlock(mid_channel2, kernel_size=kersize)
            self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, k=1)  # 🔥 修正
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)
 
    def forward(self, x):
        y = self.conv1(x)
        y = self.act(self.conv2(y))
        y = self.one_conv(y)
        y = self.act1(self.conv3(y))
        y = self.one_conv2(y)
        return y


class RepHMS(nn.Module):
    """原始RepHMS"""
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, 
                 shortcut=True, expansion=0.5, small_kersize=3, use_depthwise=True):
        super(RepHMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckUniv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)
        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)
 
    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])
        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out


# ==================== 🔥 新增：RepHMS_LALK 系列 ====================

class DepthBottleneckUni_LALK(nn.Module):
    """🔥 LALK增强的DepthBottleneck"""
    def __init__(self, in_channels, out_channels, shortcut=True, kersize=5,
                 expansion_depth=1, gate_type='spatial', use_depthwise=True):
        super(DepthBottleneckUni_LALK, self).__init__()
        
        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            
            # 🔥 LALK核心：轻量级门控
            if gate_type == 'ca':
                self.gate = CoordinateAttention(mid_channel, reduction=16)
            elif gate_type == 'spatial':
                self.gate = SpatialGate(mid_channel)
            else:
                self.gate = nn.Identity()
            
            self.one_conv = Conv(mid_channel, out_channels, k=1)  # 🔥 修正
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)
            self.gate = nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y_conv = self.act(self.conv2(y))
        y_gated = self.gate(y_conv)
        y = self.one_conv(y_gated)
        return y


class RepHMS_LALK(nn.Module):
    """🏆 RepHMS with Location-Aware Large Kernel"""
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, 
                 kersize=7, shortcut=True, expansion=0.5, gate_type='spatial'):
        super(RepHMS_LALK, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        
        self.conv1 = Conv(in_channels, c1, 1, 1)
        
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckUni_LALK(self.c_, self.c_, shortcut, kersize, 
                                        depth_expansion, gate_type=gate_type)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)
        
        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])
        
        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out


class RepHMS_LALK_Lite(nn.Module):
    """轻量级LALK版本：仅在最后一层使用门控"""
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, 
                 kersize=7, shortcut=True, expansion=0.5, gate_type='spatial'):
        super(RepHMS_LALK_Lite, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        
        self.conv1 = Conv(in_channels, c1, 1, 1)
        
        self.RepElanMSBlock = nn.ModuleList()
        for i in range(width - 1):
            DepthBlock = nn.ModuleList()
            for j in range(depth):
                if j == depth - 1:
                    DepthBlock.append(
                        DepthBottleneckUni_LALK(self.c_, self.c_, shortcut, kersize, 
                                                depth_expansion, gate_type=gate_type)
                    )
                else:
                    DepthBlock.append(
                        DepthBottleneckUniv2(self.c_, self.c_, shortcut, kersize, 
                                            depth_expansion, use_depthwise=True)
                    )
            self.RepElanMSBlock.append(DepthBlock)
        
        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])
        
        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out


# ==================== 原始ConvMS模块 ====================

class DepthBottleneckv2(nn.Module):
    """使用普通DWConv的版本"""
    def __init__(self, in_channels, out_channels, shortcut=True, kersize=5,
                 expansion_depth=1, small_kersize=3, use_depthwise=True):
        super(DepthBottleneckv2, self).__init__()
        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = DWConv(mid_channel, mid_channel, kersize)
            self.one_conv = Conv(mid_channel, mid_channel2, k=1)  # 🔥 修正
            self.conv3 = DWConv(mid_channel2, mid_channel2, kersize)
            self.one_conv2 = Conv(mid_channel2, out_channels, k=1)  # 🔥 修正
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)
 
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.one_conv(y)
        y = self.conv3(y)
        y = self.one_conv2(y)
        return y


class ConvMS(nn.Module):
    """使用普通DWConv的RepHMS版本"""
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, 
                 shortcut=True, expansion=0.5, small_kersize=3, use_depthwise=True):
        super(ConvMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)
        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)
 
    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])
        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out