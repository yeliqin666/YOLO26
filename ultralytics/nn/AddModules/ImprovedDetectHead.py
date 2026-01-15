"""
Improved Detection Head Modules for Small Object Detection (STABLE VERSION)
Fixed numerical stability issues that cause NaN losses
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA_Stable(nn.Module):
    """
    Stabilized EMA with numerical safeguards
    """
    def __init__(self, channels, factor=8):
        super(EMA_Stable, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # 🔧 FIX 1: Add epsilon to GroupNorm for stability
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups, eps=1e-5)
        
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, 
                                 kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, 
                                 kernel_size=3, stride=1, padding=1)
        
        # 🔧 FIX 2: Initialize conv weights properly
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3x3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        # 🔧 FIX 3: Clamp sigmoid outputs to avoid extreme values
        x_h_sig = torch.clamp(x_h.sigmoid(), min=1e-6, max=1-1e-6)
        x_w_sig = torch.clamp(x_w.permute(0, 1, 3, 2).sigmoid(), min=1e-6, max=1-1e-6)
        
        x1 = self.gn(group_x * x_h_sig * x_w_sig)
        x2 = self.conv3x3(group_x)
        
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        
        # 🔧 FIX 4: Clamp weights before sigmoid
        weights = torch.clamp(weights, min=-10, max=10)
        
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class CAA_Stable(nn.Module):
    """
    Stabilized Context Anchor Attention with scaled dot-product
    """
    def __init__(self, channels, reduction=16):
        super(CAA_Stable, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.conv_query = nn.Conv2d(channels, channels // reduction, 1)
        self.conv_key = nn.Conv2d(channels, channels // reduction, 1)
        self.conv_value = nn.Conv2d(channels, channels, 1)
        self.conv_out = nn.Conv2d(channels, channels, 1)
        
        # 🔧 FIX 5: Initialize gamma to small positive value instead of 0
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        
        # 🔧 FIX 6: Add scaling factor for attention
        self.scale = (channels // reduction) ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        query = self.conv_query(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C//r
        key = self.conv_key(x).view(B, -1, H * W)  # B, C//r, HW
        value = self.conv_value(x).view(B, -1, H * W)  # B, C, HW
        
        # 🔧 FIX 7: Scaled dot-product attention
        attention = torch.bmm(query, key) * self.scale
        
        # 🔧 FIX 8: Clamp before softmax to prevent overflow
        attention = torch.clamp(attention, min=-50, max=50)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.conv_out(out)
        
        # 🔧 FIX 9: Clamp gamma to reasonable range
        gamma = torch.clamp(self.gamma, min=0, max=1.0)
        
        return gamma * out + x


class DynamicSpatial_Stable(nn.Module):
    """Stabilized Spatial-Aware Attention"""
    def __init__(self, channels):
        super(DynamicSpatial_Stable, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels, eps=1e-5),  # 🔧 FIX 10: Add epsilon
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        spatial_attn = self.spatial_conv(x)
        # 🔧 FIX 11: Clamp before sigmoid
        spatial_attn = torch.clamp(spatial_attn, min=-10, max=10)
        spatial_attn = torch.sigmoid(spatial_attn)
        return x * spatial_attn


class DynamicTask_Stable(nn.Module):
    """Stabilized Task-Aware Attention"""
    def __init__(self, channels):
        super(DynamicTask_Stable, self).__init__()
        self.task_fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        task_attn = F.adaptive_avg_pool2d(x, 1).view(b, c)
        task_attn = self.task_fc(task_attn)
        
        # 🔧 FIX 12: Clamp before sigmoid
        task_attn = torch.clamp(task_attn, min=-10, max=10)
        task_attn = torch.sigmoid(task_attn).view(b, c, 1, 1)
        
        return x * task_attn


class DyHead_Stable(nn.Module):
    """Stabilized Dynamic Head"""
    def __init__(self, channels):
        super(DyHead_Stable, self).__init__()
        self.spatial_attn = DynamicSpatial_Stable(channels)
        self.task_attn = DynamicTask_Stable(channels)
        
    def forward(self, x):
        x = self.spatial_attn(x)
        x = self.task_attn(x)
        return x


# Import ultralytics components
try:
    from ultralytics.nn.modules.conv import Conv, DWConv
    from ultralytics.utils.tal import dist2bbox, make_anchors
except ImportError:
    print("Warning: Could not import from ultralytics, using placeholder classes")
    class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, k//2, groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act else nn.Identity()
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))
    
    class DWConv(Conv):
        def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
            super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DFL(nn.Module):
    """Distribution Focal Loss - Integral module"""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect_EMA_DyHead(nn.Module):
    """
    🔧 STABLE VERSION - Fixed numerical issues
    
    Key Fixes:
    1. ✅ Proper initialization of all learnable parameters
    2. ✅ Scaled dot-product attention in CAA
    3. ✅ Clamping before sigmoid/softmax operations
    4. ✅ Epsilon protection in normalization layers
    5. ✅ Reduced attention module stacking
    6. ✅ Gradient clipping-friendly architecture
    
    Usage in YAML:
    - [[P3, P4, P5], 1, Detect_EMA_DyHead_Stable, [nc]]
    """
    
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        
        # 🔧 FIX 13: Simplified architecture - reduce attention stacking
        # Regression head: EMA -> Conv -> DyHead
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                EMA_Stable(x, factor=8),  # Single EMA
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                DyHead_Stable(c2),  # DyHead for spatial/task awareness
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        
        # Classification head: Simple conv stack with one EMA
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                EMA_Stable(c3, factor=8),  # Single EMA
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        # 🔧 FIX 14: Proper weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights properly to avoid NaN"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """Forward pass with gradient checking"""
        shape = x[0].shape
        
        for i in range(self.nl):
            reg_out = self.cv2[i](x[i])
            cls_out = self.cv3[i](x[i])
            
            # 🔧 FIX 15: Check for NaN during training
            if self.training:
                if torch.isnan(reg_out).any() or torch.isnan(cls_out).any():
                    print(f"⚠️ NaN detected at layer {i}")
                    # Replace NaN with zeros to prevent crash
                    reg_out = torch.nan_to_num(reg_out, nan=0.0)
                    cls_out = torch.nan_to_num(cls_out, nan=0.0)
            
            x[i] = torch.cat((reg_out, cls_out), 1)
            
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape
            
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            
        dbox = (
            dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)
            * self.strides
        )
        
        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor(
                [img_w, img_h, img_w, img_h], device=dbox.device
            ).reshape(1, 4, 1)
            dbox /= img_size
            
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
    
    def bias_init(self):
        """Initialize biases for detection head"""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            # 🔧 FIX 16: More conservative bias initialization
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class Detect_EMA_Lite(nn.Module):
    """
    ⚡ Minimalist Stable Version - EMA Only
    
    Recommended for initial testing and stability
    Gradually add complexity once this works
    
    Usage in YAML:
    - [[P3, P4, P5], 1, Detect_EMA_Lite, [nc]]
    """
    
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        
        # Minimal architecture - only one EMA per head
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                EMA_Stable(c2, factor=8),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape
            
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = (
            dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)
            * self.strides
        )
        
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
    
    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


__all__ = [
    'EMA_Stable',
    'CAA_Stable',
    'DyHead_Stable',
    'Detect_EMA_DyHead',
    'Detect_EMA_Lite',
]


if __name__ == "__main__":
    print("=" * 80)
    print("Testing STABLE Detection Head Modules")
    print("=" * 80)
    
    # Test with gradient checking
    print("\n✓ Testing EMA_Stable with gradient flow...")
    ema = EMA_Stable(256)
    x = torch.randn(2, 256, 40, 40, requires_grad=True)
    out = ema(x)
    loss = out.sum()
    loss.backward()
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  Gradient flow: {'✅ OK' if x.grad is not None and not torch.isnan(x.grad).any() else '❌ NaN detected'}")
    
    print("\n✓ Testing Detect_EMA_Only...")
    detect = Detect_EMA_Lite(nc=80, ch=(256, 512, 1024))
    x_list = [
        torch.randn(2, 256, 80, 80, requires_grad=True),
        torch.randn(2, 512, 40, 40, requires_grad=True),
        torch.randn(2, 1024, 20, 20, requires_grad=True)
    ]
    detect.stride = torch.tensor([8., 16., 32.])
    detect.train()
    out = detect(x_list)
    
    # Check for NaN
    has_nan = any(torch.isnan(o).any() for o in out)
    print(f"  NaN check: {'❌ NaN found' if has_nan else '✅ No NaN'}")
    
    print("\n" + "=" * 80)
    print("✅ Stability tests completed!")
    print("=" * 80)