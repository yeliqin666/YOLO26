import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :]
        x02 = x[:, :, 1::2, :]
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        
        # Haar小波
        ll = (x1 + x2 + x3 + x4) * 0.5
        lh = (-x1 - x2 + x3 + x4) * 0.5
        hl = (-x1 + x2 - x3 + x4) * 0.5
        hh = (x1 - x2 - x3 + x4) * 0.5
        
        return torch.cat([ll, lh, hl, hh], dim=1)

class IWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        C_out = C // 4
        
        ll = x[:, 0:C_out, :, :]
        lh = x[:, C_out:C_out*2, :, :]
        hl = x[:, C_out*2:C_out*3, :, :]
        hh = x[:, C_out*3:C_out*4, :, :]
        
        h = torch.zeros([B, C_out, H*2, W*2], device=x.device, dtype=x.dtype)
        h[:, :, 0::2, 0::2] = ll - lh - hl + hh
        h[:, :, 1::2, 0::2] = ll - lh + hl - hh
        h[:, :, 0::2, 1::2] = ll + lh - hl - hh
        h[:, :, 1::2, 1::2] = ll + lh + hl + hh
        
        return h

class LowFreqSparseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, k_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.k_ratio = k_ratio
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, dim//4), num_channels=dim, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Q, K 归一化 (Cosine Attention 变体)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # =======================================================
        # 优化 1: 动态 Clamp
        # =======================================================
        # FP16 最大表示 65504, exp(11.1) ≈ 65500。
        # FP32 虽然可以很大，但 Softmax 输入过大会导致梯度消失。
        # 这里根据 dtype 动态调整阈值，既不溢出又尽可能放宽。
        if attn.dtype == torch.float16:
            clamp_val = 10.0 # 安全区
        else:
            clamp_val = 32.0 # 激进区
            
        attn = torch.clamp(attn, min=-clamp_val, max=clamp_val)
        
        # =======================================================
        # 优化 2: 高效 Top-K Masking
        # =======================================================
        k_num = max(int(N * self.k_ratio), 1)
        if k_num < N:
            # 这里的 topk 依然需要计算，但后续掩码生成更高效
            _, topk_idx = torch.topk(attn, k_num, dim=-1)
            
            # 使用 bool mask + where，比原来的乘法掩码更节约显存且更清晰
            mask = torch.zeros_like(attn, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            
            # 填充负无穷。FP16 下 -1e9 会变成 -inf，但为了安全我们用 -65500
            fill_value = -65500.0 if attn.dtype == torch.float16 else -1e9
            attn = torch.where(mask, attn, torch.tensor(fill_value, device=attn.device, dtype=attn.dtype))
        
        # Softmax 能够处理 max-subtraction，只要输入本身不是 inf 就不会 NaN
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v)
        out = out.transpose(2, 3).reshape(B, C, H, W)
        out = self.proj(out)
        out = self.norm(out)
        return out

class HighFreqEnhancer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        high_dim = dim * 3
        num_groups = min(32, high_dim // 8)
        
        self.enhance = nn.Sequential(
            nn.Conv2d(high_dim, high_dim, 3, 1, 1, groups=high_dim, bias=False),
            nn.GroupNorm(num_groups, high_dim, eps=1e-6),
            nn.SiLU(inplace=True),
            nn.Conv2d(high_dim, high_dim, 1, bias=False),
            nn.GroupNorm(num_groups, high_dim, eps=1e-6),
        )
        self.alpha = nn.Parameter(torch.tensor([0.1])) 
    
    def forward(self, x):
        return x + self.alpha * self.enhance(x)

class WaveletSparseBlock(nn.Module):
    def __init__(self, dim, num_heads=8, k_ratio=0.5, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.dwt = DWT()
        self.iwt = IWT()
        
        self.low_attn = LowFreqSparseAttention(dim, num_heads, k_ratio)
        self.high_enhance = HighFreqEnhancer(dim)
        
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, dim // 4), num_channels=dim * 2, eps=1e-6),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, dim // 4), num_channels=dim, eps=1e-6),
        )
        
        self.layer_scale_1 = 0.1
        self.layer_scale_2 = 0.1
    
    def _forward_impl(self, x):
        # 移除了所有强制类型转换，直接计算
        
        # 1. 小波分解
        x_dwt = self.dwt(x)
        B, C_in, H, W = x.shape
        ll = x_dwt[:, :C_in, :, :]
        high = x_dwt[:, C_in:, :, :]
        
        # 2. 分支处理
        ll_out = self.low_attn(ll)
        high_out = self.high_enhance(high)
        
        # 3. 重构
        x_rec = self.iwt(torch.cat([ll + ll_out, high_out], dim=1))
        
        # 4. 残差连接 1
        out = x + self.layer_scale_1 * x_rec
        
        # 5. FFN
        out = out + self.layer_scale_2 * self.ffn(out)
        
        return out
    
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

class A2C2f_WaveletSparse(nn.Module):
    """
    轻量化版本: 移除了防御性 FP32 转换，依赖动态 Clamp 保证安全
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, use_checkpoint=False):
        super().__init__()
        c_ = int(c2 * e)
        
        self.cv1 = nn.Conv2d(c1, 2 * c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * c_, eps=1e-5)
        
        self.m = nn.ModuleList(
            WaveletSparseBlock(
                c_, 
                num_heads=max(1, c_ // 32),
                k_ratio=0.5,
                use_checkpoint=use_checkpoint
            ) 
            for _ in range(n)
        )
        
        self.cv2 = nn.Conv2d((2 + n) * c_, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2, eps=1e-5)
        
        # 保留梯度裁剪，这几乎没有开销但对稳定性极好
        self._register_gradient_clipping()

    def _register_gradient_clipping(self):
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: torch.clamp(grad, -5.0, 5.0))

    def forward(self, x):
        # 移除了所有的 cast 和 autocast(False)
        # 纯净的 Forward Pass
        x_proj = self.bn1(self.cv1(x))
        y = list(x_proj.chunk(2, 1))
        
        for m in self.m:
            y.append(m(y[-1]))
        
        out = torch.cat(y, 1)
        out = self.bn2(self.cv2(out))
        
        return out