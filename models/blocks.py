"""
models/blocks.py
=================
Transformer 基础构件模块。

来源说明：
  - RMSNorm、SwiGLU、JiTAttention、JiTBlock 参考
    Just-Image-Transformers-main/src/nanojit/model.py
  - 原始实现针对图像 patch 序列；本文件已适配为通用序列（变长点集）
  - 新增：padding mask 支持（attn_mask 参数），屏蔽 padding 点对注意力的影响
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ======================================================================
# 基础归一化
# ======================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization（参考 JustImageTransformer）"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps   = eps
        self.g     = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g


# ======================================================================
# 前馈网络
# ======================================================================

class SwiGLU(nn.Module):
    """
    Gated Linear Unit with SiLU 激活（参考 JustImageTransformer）。
    比标准 MLP 有更好的表达能力。
    公式：output = w3(SiLU(w1(x)) * w2(x))
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1      = nn.Linear(dim, hidden_dim, bias=False)
        self.w2      = nn.Linear(dim, hidden_dim, bias=False)
        self.w3      = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ======================================================================
# 旋转位置编码（RoPE）
# ======================================================================

def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    将旋转位置编码应用到 Q 或 K。
    参考 JustImageTransformer，保持原始实现不变。

    Args:
        x:     [B, N, H, D/2*2]  Q 或 K，维度需为偶数
        freqs: [1, N_max, 1, D/4]  预计算的复数频率（取前 N 个）
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # 取对应序列长度的频率
    freqs_used = freqs[:, :x.shape[1], :, :]
    x_out = torch.view_as_real(x_complex * freqs_used).flatten(3)
    return x_out.type_as(x)


# ======================================================================
# 注意力模块
# ======================================================================

class STECAttention(nn.Module):
    """
    多头自注意力（Flash Attention 加速 + padding 置零）。

    使用 PyTorch 2.0+ 的 F.scaled_dot_product_attention，不传 attn_mask，
    让 SDPA 自动选择 Flash Attention 后端（sm_80+ GPU）。
    Padding 处理方式：在调用 SDPA 前将 padding 位置的 Q/K/V 置零，
    使 padding key 不贡献有效注意力信号。

    Args:
        dim:     模型维度
        heads:   注意力头数
        dropout: attention dropout 概率
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim 必须能被 heads 整除"
        self.heads    = heads
        self.head_dim = dim // heads

        self.qkv    = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj   = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, N, dim]
            freqs:    [1, N, 1, head_dim/2]  RoPE 频率
            pad_mask: [B, N]  bool，True 表示该位置是 padding（需屏蔽）

        Returns:
            out: [B, N, dim]
        """
        B, N, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        q = q.transpose(1, 2)  # [B, H, N, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Padding 屏蔽：给 SDPA 传 attn_mask，softmax 对 padding key 完全不分配权重。
        # 相比"置零 K/V 不传 mask"的旧方案，这里彻底消除 padding 数量差异导致的
        # 注意力稀释偏差，使 B>1 打包推理与 B=1 逐样本推理数值一致。
        # 代价：传 attn_mask 时 SDPA 回退到 MATH/Efficient 后端（非 Flash），
        # 但保证了不同 batch 组合下结果不变，精度优先。
        attn_mask = None
        if pad_mask is not None:
            # key 维屏蔽：padding key 不被任何 query attend
            # [B, N] -> [B, 1, 1, N]，广播到 [B, H, N, N]
            attn_mask = (~pad_mask)[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [B, H, N, D_h]

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


# ======================================================================
# Transformer Block（带 AdaLN 条件调制）
# ======================================================================

class STECBlock(nn.Module):
    """
    单个 Transformer Block，带 Adaptive Layer Normalization（AdaLN）。

    AdaLN 来源：JustImageTransformer JiTBlock，用于将扩散时间步和条件信息
    注入到每个 block，调制特征的 scale 和 shift。

    本 block 与原始实现的差异：
      - 支持 pad_mask（padding 点屏蔽）
      - 条件向量 cond 包含时间步嵌入 + context 聚合嵌入

    Args:
        dim:       模型维度
        heads:     注意力头数
        mlp_ratio: FFN 维度扩展比例
        dropout:   dropout 概率
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = STECAttention(dim, heads, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio * 2 / 3)  # SwiGLU 建议维度
        self.mlp   = SwiGLU(dim, hidden_dim, dropout=dropout)

        # AdaLN：从条件向量预测 6 个调制参数
        # [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # 初始化为 0，确保训练初期 block 相当于恒等映射
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        freqs: torch.Tensor,
        pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, N, dim]   输入特征
            cond:     [B, dim]      条件向量（时间步 + context 聚合）
            freqs:    [1, N, 1, head_dim/2]  RoPE 频率
            pad_mask: [B, N]  bool，True 为 padding

        Returns:
            x: [B, N, dim]
        """
        # AdaLN 调制参数
        ada = self.adaLN(cond)  # [B, 6*dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            ada.chunk(6, dim=1)  # 各 [B, dim]

        # Self-Attention 分支（Pre-Norm + AdaLN调制）
        x_normed = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_normed, freqs, pad_mask)

        # FFN 分支（Pre-Norm + AdaLN调制）
        x_normed = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_normed)

        return x
