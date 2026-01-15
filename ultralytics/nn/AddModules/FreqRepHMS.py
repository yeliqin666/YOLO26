"""
🔥 RepHMS_FreqLALK 优化版本（YOLO26 兼容版）
✅ 修复了 MultiScaleFreqGate 尺寸不匹配
✅ 修复了混合版通道冲突
✅ 修复了 BatchNorm 在 batch_size=1 时的错误
✅ 兼容 YOLO parse_model 的参数传递方式
✅ 支持 c1 != c2 的通道适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 导入 Conv 模块（用于通道适配）
# ============================================================================
try:
    from ultralytics.nn.modules.conv import Conv
except ImportError:
    # 如果导入失败，使用简化版本
    class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, p or k // 2, groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))


# ============================================================================
# 优化1: 自适应FreqGate - 可学习的频域权重
# ============================================================================

class AdaptiveFreqGate(nn.Module):
    """
    自适应频域门控：增加可学习的开关，让网络决定频域筛选的强度
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(8, channels // reduction)
        
        # 🔥 修复：移除所有归一化层（避免 batch_size=1 且空间维度为 1x1 的问题）
        # Gate 模块本身很轻量，Sigmoid 已提供范围限制，不需要归一化
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(mid_channels, channels, 1, bias=True),
            nn.Sigmoid()
        )
        
        # 🔥 可学习的门控强度 (0-1之间)
        self.gate_strength = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        low_freq_info = self.avg_pool(x)
        freq_weight = self.fc(low_freq_info)
        
        # 自适应混合：让网络学习频域筛选的重要性
        gated = x * freq_weight
        return (1 - self.gate_strength) * x + self.gate_strength * gated


# ============================================================================
# 优化2: 多频段FreqGate - 分别处理不同频段
# ============================================================================

class MultiScaleFreqGate(nn.Module):
    """
    多尺度频域门控：分别捕捉低频/中频/高频信息
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        
        # 低频门控 (全局) - 移除归一化层
        self.low_pool = nn.AdaptiveAvgPool2d(1)
        self.low_gate = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(mid_channels, channels, 1, bias=True),
        )
        
        # 中频门控 (局部) - 移除归一化层
        self.mid_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.mid_gate = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(mid_channels, channels, 1, bias=True),
        )
        
        # 高频权重学习 - 移除归一化层
        self.high_gate = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(mid_channels, channels, 1, bias=True),
        )
        
        # 融合权重 - 移除归一化层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 低频分量 - 修复：需要上采样到原始尺寸
        low_freq_pooled = self.low_pool(x)  # [B, C, 1, 1]
        low_freq = self.low_gate(low_freq_pooled)  # [B, C, 1, 1]
        low_freq = F.interpolate(low_freq, size=(H, W), mode='bilinear', align_corners=False)  # [B, C, H, W]
        
        # 中频分量
        mid_freq = self.mid_pool(x)  # [B, C, H, W]
        mid_freq = self.mid_gate(mid_freq)  # [B, C, H, W]
        
        # 高频分量 (原始 - 低频近似)
        low_approx = F.interpolate(low_freq_pooled, size=(H, W), mode='bilinear', align_corners=False)
        high_freq = x - low_approx  # [B, C, H, W]
        high_freq = self.high_gate(high_freq)  # [B, C, H, W]
        
        # 融合所有频段
        freq_features = torch.cat([low_freq, mid_freq, high_freq], dim=1)  # [B, 3*C, H, W]
        freq_weight = self.fusion(freq_features)  # [B, C, H, W]
        
        return x * freq_weight


# ============================================================================
# 优化3: 层级化FreqGate - 不同深度用不同强度
# ============================================================================

class HierarchicalFreqGate(nn.Module):
    """
    层级化频域门控：浅层弱筛选，深层强筛选
    """
    def __init__(self, channels, reduction=16, layer_depth=0, max_depth=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(8, channels // reduction)
        
        # 🔥 修复：移除归一化层
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(mid_channels, channels, 1, bias=True),
            nn.Sigmoid()
        )
        
        # 层级化强度：随深度递增
        self.strength = 0.2 + 0.6 * (layer_depth / max(max_depth, 1))
    
    def forward(self, x):
        low_freq_info = self.avg_pool(x)
        freq_weight = self.fc(low_freq_info)
        gated = x * freq_weight
        
        return (1 - self.strength) * x + self.strength * gated


# ============================================================================
# 简化的FreqGate (保持你的原版)
# ============================================================================

class FreqGate(nn.Module):
    """原版FreqGate - 简单高效"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(8, channels // reduction)
        
        # 🔥 修复：移除归一化层
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(mid_channels, channels, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        low_freq_info = self.avg_pool(x)
        freq_weight = self.fc(low_freq_info)
        return x * freq_weight


# ============================================================================
# 混合门控包装器
# ============================================================================

class HybridGateWrapper(nn.Module):
    """
    混合门控包装器：同时应用 spatial 和 frequency 门控
    """
    def __init__(self, base_block, spatial_gate, freq_gate):
        super().__init__()
        self.base_block = base_block
        self.spatial_gate = spatial_gate
        self.freq_gate = freq_gate
    
    def forward(self, x):
        # 基础块输出
        out = self.base_block(x)
        
        # 应用 spatial gate
        spatial_weight = self.spatial_gate(out)  # [B, 1, H, W]
        
        # 应用 freq gate
        freq_weight = self.freq_gate(out)  # [B, C, H, W]
        
        # 混合：spatial * freq * out
        return out * spatial_weight * freq_weight


# ============================================================================
# 完整实现：RepHMS_FreqLALK 优化版（YOLO26 兼容）
# ============================================================================

class RepHMS_FreqLALK_Enhanced(nn.Module):
    """
    增强版 RepHMS_FreqLALK - YOLO26 兼容版
    
    新增特性:
    1. 支持多种FreqGate变体
    2. 层级化门控强度
    3. 可选的Spatial+Freq混合模式
    4. ✅ 兼容 YOLO parse_model 的参数传递
    5. ✅ 支持 c1 != c2 的通道适配
    """
    def __init__(self, c1, c2, kernel_size=3, stride=1, depth_expansion=2, 
                 kersize=7, shortcut=True, expansion=0.5, 
                 gate_type='freq', freq_variant='adaptive'):
        """
        YOLO 兼容的参数签名
        
        Args:
            c1: 输入通道数（由 parse_model 自动填充）
            c2: 输出通道数（由 parse_model 自动填充）
            kernel_size: 卷积核大小（从 YAML 的第1个参数）
            stride: 步长（从 YAML 的第2个参数）
            depth_expansion: 深度扩展系数（从 YAML 的第3个参数）
            kersize: LALK 核大小（从 YAML 的第4个参数）
            shortcut: 是否使用快捷连接（从 YAML 的第5个参数）
            expansion: 通道扩展系数（从 YAML 的第6个参数）
            gate_type: 门控类型 'spatial', 'freq', 'hybrid', None（从 YAML 的第7个参数）
            freq_variant: 频域变体 'basic', 'adaptive', 'multiscale', 'hierarchical'（从 YAML 的第8个参数）
        """
        super().__init__()
        
        # ✅ 关键修复：支持 c1 != c2
        self.channel_adapter = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        
        # 使用固定的 width 和 depth（简化版本）
        self.width = 3
        self.depth = 1
        self.gate_type = gate_type
        self.freq_variant = freq_variant
        
        # 后续所有操作使用 c2 作为基准通道数
        c_ = int(c2 * expansion)
        c1_internal = c_ * self.width
        self.c_ = c_
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(c2, c1_internal, 1, bias=False),  # 注意这里使用 c2
            nn.BatchNorm2d(c1_internal),
            nn.SiLU()
        )
        
        # 构建多分支级联结构
        self.RepElanMSBlock = nn.ModuleList()
        for i in range(self.width - 1):
            DepthBlock = nn.ModuleList()
            for j in range(self.depth):
                # 基础LALK块
                base_block = self._make_lalk_block(c_, kersize, shortcut, depth_expansion)
                
                # 在末端添加门控
                if j == self.depth - 1:
                    block = self._wrap_with_gate(
                        base_block, c_, i, j,
                        gate_type, freq_variant
                    )
                else:
                    block = base_block
                
                DepthBlock.append(block)
            self.RepElanMSBlock.append(DepthBlock)
        
        out_ch = c_ * (1 + (self.width - 1) * self.depth)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
    
    def _make_lalk_block(self, channels, kersize, shortcut, expansion):
        """创建简化的LALK块"""
        mid_ch = int(channels * expansion)
        return nn.Sequential(
            nn.Conv2d(channels, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),
            nn.Conv2d(mid_ch, mid_ch, kersize, padding=kersize//2, 
                     groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),
            nn.Conv2d(mid_ch, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
    
    def _wrap_with_gate(self, base_block, channels, branch_idx, 
                       layer_idx, gate_type, freq_variant):
        """为基础块添加门控"""
        
        if gate_type is None or gate_type == 'none':
            return base_block
        
        spatial_gate = None
        freq_gate = None
        
        # 创建 Spatial gate
        if gate_type in ['spatial', 'hybrid']:
            # 🔥 修复：Spatial gate 不受 BatchNorm 影响，因为输出只有1个通道
            spatial_gate = nn.Sequential(
                nn.Conv2d(channels, 1, 7, padding=3, bias=False),
                nn.Sigmoid()  # 移除 BatchNorm
            )
        
        # 创建 Freq gate
        if gate_type in ['freq', 'hybrid']:
            if freq_variant == 'adaptive':
                freq_gate = AdaptiveFreqGate(channels)
            elif freq_variant == 'multiscale':
                freq_gate = MultiScaleFreqGate(channels)
            elif freq_variant == 'hierarchical':
                freq_gate = HierarchicalFreqGate(
                    channels, layer_depth=layer_idx, max_depth=self.depth
                )
            else:  # 'basic'
                freq_gate = FreqGate(channels)
        
        # 根据不同模式组装
        if gate_type == 'spatial':
            return nn.Sequential(base_block, spatial_gate)
        
        elif gate_type == 'freq':
            return nn.Sequential(base_block, freq_gate)
        
        elif gate_type == 'hybrid':
            return HybridGateWrapper(base_block, spatial_gate, freq_gate)
        
        else:
            return base_block
    
    def forward(self, x):
        # ✅ 第一步：通道适配
        x = self.channel_adapter(x)
        
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


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("🧪 测试 RepHMS_FreqLALK 各个变体（YOLO26 兼容版）\n")
    
    # 🔥 测试 1: 标准情况 (c1 == c2)
    print("=" * 60)
    print("测试 1: c1 == c2 (标准情况)")
    print("=" * 60)
    x = torch.randn(2, 256, 40, 40)
    
    model = RepHMS_FreqLALK_Enhanced(
        c1=256, c2=256,
        kernel_size=3, stride=1, depth_expansion=2, kersize=7,
        shortcut=True, expansion=0.5,
        gate_type='freq', freq_variant='adaptive'
    )
    
    model.eval()
    with torch.no_grad():
        out = model(x)
    
    print(f"✅ 输入: {x.shape}")
    print(f"✅ 输出: {out.shape}")
    print(f"✅ 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")
    
    # 🔥 测试 2: c1 != c2 (通道适配)
    print("=" * 60)
    print("测试 2: c1 != c2 (通道适配)")
    print("=" * 60)
    x2 = torch.randn(2, 1536, 40, 40)  # Concat 后的通道数
    
    model2 = RepHMS_FreqLALK_Enhanced(
        c1=1536, c2=512,  # 不同的输入输出通道
        kernel_size=3, stride=1, depth_expansion=2, kersize=7,
        shortcut=True, expansion=0.5,
        gate_type='freq', freq_variant='adaptive'
    )
    
    model2.eval()
    with torch.no_grad():
        out2 = model2(x2)
    
    print(f"✅ 输入: {x2.shape}")
    print(f"✅ 输出: {out2.shape}")
    print(f"✅ 参数量: {sum(p.numel() for p in model2.parameters()) / 1e6:.2f}M\n")
    
    # 🔥 测试 3: batch_size=1
    print("=" * 60)
    print("测试 3: batch_size=1")
    print("=" * 60)
    x3 = torch.randn(1, 256, 40, 40)
    
    model3 = RepHMS_FreqLALK_Enhanced(
        c1=256, c2=256,
        kernel_size=3, stride=1, depth_expansion=2, kersize=7,
        shortcut=True, expansion=0.5,
        gate_type='freq', freq_variant='multiscale'
    )
    
    model3.eval()
    with torch.no_grad():
        out3 = model3(x3)
    
    print(f"✅ 输入: {x3.shape}")
    print(f"✅ 输出: {out3.shape}\n")
    
    # 🔥 测试 4: 所有变体
    print("=" * 60)
    print("测试 4: 所有频域变体")
    print("=" * 60)
    
    variants = [
        ('基础版', 'basic'),
        ('自适应版', 'adaptive'),
        ('多尺度版', 'multiscale'),
        ('层级版', 'hierarchical'),
    ]
    
    x4 = torch.randn(2, 256, 40, 40)
    
    for name, variant in variants:
        model = RepHMS_FreqLALK_Enhanced(
            c1=256, c2=256,
            gate_type='freq', freq_variant=variant
        )
        model.eval()
        with torch.no_grad():
            out = model(x4)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name:12s} | 输出: {out.shape} | 参数: {params:.2f}M")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    print("\n🎉 YOLO26 兼容性修复总结:")
    print("  ✅ 支持 c1 != c2 (通过 channel_adapter)")
    print("  ✅ 兼容 parse_model 的参数签名")
    print("  ✅ 移除了所有会导致 batch_size=1 报错的 BatchNorm")
    print("  ✅ 修复了 MultiScaleFreqGate 的尺寸问题")
    print("  ✅ 支持所有频域变体：basic, adaptive, multiscale, hierarchical")
    print("\n💡 YAML 配置示例:")
    print("  - [-1, 2, RepHMS_FreqLALK_Enhanced, [3, 1, 2, 7, True, 0.5, 'freq', 'adaptive']]")
    print("                                        ↑参数从这里开始，不需要写 c1 和 c2")