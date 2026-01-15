# ----------------------------------------------------
# ⬇️ 这是一个完整的、修复了所有Bug的 SAMamba.py 文件 ⬇️
# ----------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 警告：此文件依赖 mamba_ssm。
# 如果您只想用 DPCF/MDCR，请手动删除 'from mamba_ssm' 
# 以及 CSI 和 MAMBACR 模块。
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None  # 允许在没有 Mamba 的情况下导入

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

# --- 依赖模块: conv_block ---
class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

# --- 依赖模块: AdaptiveCombiner ---
class AdaptiveCombiner(nn.Module):
    def __init__(self):
        super(AdaptiveCombiner, self).__init__()
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, p, i):
        batch_size, channel, w, h = p.shape
        d = self.d.expand(batch_size, channel, w, h)
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i

# --- 依赖模块: Bag ---
class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i

# --- 依赖模块: ECA ---
class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

# --- 核心模块 1: DPCF (已修复) ---
class DPCF(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        # DPCF 将输入分成 4 块，所以 in_features 应该是 c1 (来自 x_low/x_high)
        # 它将 4 块融合后 cat, 所以 tail_conv 的 in_features 应该是 c1
        # 但 tail_conv 的 out_features 是 yaml 里的 c2
        self.ac = AdaptiveCombiner()
        self.tail_conv = nn.Sequential(
            conv_block(in_features=in_features,  # 这里的 in_features 是 c1 (例如 512)
                       out_features=out_features, # 这里的 out_features 是 c2 (例如 1024)
                       kernel_size=(1, 1),
                       padding=(0, 0))
        )

    def forward(self, x):
        """
        DPCF 模块的前向传播 (最终修正版 - 修复矩形Bug)
        x: 是一个包含两个张量的列表 [x_low, x_high]
        """
        x_low, x_high = x
        target_shape = x_high.shape[2:] # 获取 [H, W]

        x_low_interpolated = F.interpolate(x_low,
                                             size=target_shape,
                                             mode='bilinear',
                                             align_corners=True)

        if x_low_interpolated.shape != x_high.shape:
            raise ValueError(f"DPCF shape mismatch: x_low {x_low_interpolated.shape} vs x_high {x_high.shape}")

        x_low_chunks = torch.chunk(x_low_interpolated, 4, dim=1)
        x_high_chunks = torch.chunk(x_high, 4, dim=1)

        x0 = self.ac(x_low_chunks[0], x_high_chunks[0])
        x1 = self.ac(x_low_chunks[1], x_high_chunks[1])
        x2 = self.ac(x_low_chunks[2], x_high_chunks[2])
        x3 = self.ac(x_low_chunks[3], x_high_chunks[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x

# --- 核心模块 2: MDCR (已修复) ---
class MDCR(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()

        # --- ⬇️ 这是“最终修复”代码 ⬇️ ---
        # 1. 检查通道是否能被 4 整除
        if in_features % 4 != 0 or out_features % 4 != 0:
            raise ValueError(f"MDCR: in_features({in_features}) and out_features({out_features}) must be divisible by 4.")
        
        c_in_block = in_features // 4
        c_out_block = out_features // 4

        # 2. 动态计算 'groups'
        #    原硬编码 groups=128, 是基于 in_features=1024 (1024//8=128)
        #    我们动态计算 groups = in_features // 8
        dynamic_groups = max(1, in_features // 8)

        # 3. 再次检查 c_in_block (in_channels) 是否能被 dynamic_groups 整除
        if c_in_block % dynamic_groups != 0:
            # 这是一个备用保险，如果 1024/8 的逻辑也错了
            print(f"MDCR Warning: Fallback groups. {c_in_block} is not divisible by {dynamic_groups}.")
            dynamic_groups = 1 # 退回到标准卷积
        # --- ⬆️ 修复结束 ⬆️ ---

        self.block1 = conv_block(
            in_features=c_in_block,
            out_features=c_out_block,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups = dynamic_groups  # ⬅️ 使用动态 groups
            )
        self.block2 = conv_block(
            in_features=c_in_block,
            out_features=c_out_block,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups = dynamic_groups  # ⬅️ 使用动态 groups
            )
        self.block3 = conv_block(
            in_features=c_in_block,
            out_features=c_out_block,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation,
            groups = dynamic_groups  # ⬅️ 使用动态 groups
            )
        self.block4 = conv_block(
            in_features=c_in_block,
            out_features=c_out_block,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation,
            groups = dynamic_groups  # ⬅️ 使用动态 groups
            )
        self.out_s = conv_block(
            in_features=4,
            out_features=4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )

    def forward(self, x):
        # 1. 将输入分块
        x = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        x3 = self.block3(x[2])
        x4 = self.block4(x[3])

        # --- ⬇️ 性能优化：向量化循环 ⬇️ ---
        # 原始的 'for channel in range(x1.size(1))' 循环非常慢
        
        B, C_block, H, W = x1.shape
        
        # 1. 堆叠: [B, C_block, H, W] * 4 -> [B, 4, C_block, H, W]
        x_stacked = torch.stack([x1, x2, x3, x4], dim=1)
        
        # 2. 换轴: -> [B, C_block, 4, H, W]
        x_permuted = x_stacked.permute(0, 2, 1, 3, 4)
        
        # 3. 重塑以进行 out_s (in_channels=4)
        #    -> [ (B * C_block), 4, H, W ]
        x_reshaped = x_permuted.reshape(B * C_block, 4, H, W)
        
        # 4. 应用 out_s 卷积
        out_s_applied = self.out_s(x_reshaped)
        
        # 5. 恢复形状 (C_block * 4 = out_features)
        #    -> [ B, C_block, 4, H, W ]
        out_unshaped = out_s_applied.reshape(B, C_block, 4, H, W)
        #    -> [ B, (C_block * 4), H, W ]
        x = out_unshaped.reshape(B, C_block * 4, H, W)

        # --- ⬆️ 向量化结束 ⬆️ ---

        x = self.out(x)
        return x

# --- 核心模块 3: CSI (依赖 Mamba) ---
class CSI(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba-ssm is not installed. CSI module cannot be used.")

        self.skip = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type='bn',
                                activation=True)
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.drop = nn.Dropout2d(0.3)
        self.mambacr = MAMBACR(filters, filters)
        self.final_conv = conv_block(in_features=filters,
                                        out_features=filters,
                                        kernel_size=(1, 1),
                                        padding=(0, 0))

    def forward(self, x):
        x_skip = self.skip(x)
        x = self.mambacr(x_skip)
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.final_conv(x_skip + x)
        return x

# --- 核心模块 4: MAMBACR (依赖 Mamba) ---
class MAMBACR(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba-ssm is not installed. MAMBACR module cannot be used.")
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm1 = nn.LayerNorm(input_dim//4)
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim//4, output_dim//4)
        self.skip_scale= nn.Parameter(torch.ones(1))
        self.cpe2 = nn.Conv2d(input_dim//4, input_dim//4, 3, padding=1, groups=input_dim//4)
        self.out = conv_block(
            in_features=output_dim,
            out_features=output_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        self.mlp = Mlp(in_features=input_dim//4, hidden_features=int(input_dim//4 * 4))
        
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, H, W = x.shape
        assert C == self.input_dim
        n_tokens = H * W
        img_dims = (H, W)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mlp(self.norm1(self.mamba(x1))) + self.skip_scale * x1
        x_mamba2 = self.mlp(self.norm1(self.mamba(x2))) + self.skip_scale * x2
        x_mamba3 = self.mlp(self.norm1(self.mamba(x3))) + self.skip_scale * x3
        x_mamba4 = self.mlp(self.norm1(self.mamba(x4))) + self.skip_scale * x4

        x_mamba1 = x_mamba1.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)
        x_mamba2 = x_mamba2.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)
        x_mamba3 = x_mamba3.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)
        x_mamba4 = x_mamba4.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)

        # 向量化循环
        x_stacked = torch.stack([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=1) # [B, 4, C_block, H, W]
        x_permuted = x_stacked.permute(0, 2, 1, 3, 4) # [B, C_block, 4, H, W]
        
        # 注意: 原代码的 out_s 在这里没有被使用, 而是直接 cat
        # 我们遵循原代码的逻辑
        
        # x = torch.cat(split_tensors, dim=1) # 原代码的逻辑
        # x_mamba1...x_mamba4 是 [B, C_out_block, H, W]
        # (假设 output_dim == input_dim)
        x = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=1)
        
        out = self.out(x)
        return out