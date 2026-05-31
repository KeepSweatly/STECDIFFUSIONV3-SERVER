"""
models/transformer.py
======================
STEC 条件扩散噪声预测网络（多星联合版本）

概述：
    本模块实现基于 Transformer 的离散点集 STEC 噪声预测网络，用于条件扩散模型
    的去噪过程。网络设计参考 JustImageTransformer（JiT），适配为离散点任务。

核心架构：STECDiffTransformer
    输入特征：
        - noisy_stec:    [B, N, 1]   加噪后的 STEC（target 处是 xt，context 处是原始值）
        - coords:        [B, N, 2]   归一化经纬度（lat, lon）
        - angles:        [B, N, 2]   归一化角度（azimuth, elevation）
        - system_ids:    [B, N]      卫星系统 ID（0=padding, 1=GPS, 2=GLONASS, ...）
        - context_stec:  [B, N, 1]   context 处真实 STEC，target 和 padding 处为 0
        - role_type:     [B, N]      角色类型（0=padding, 1=context, 2=target）
        - valid_mask:    [B, N]      bool，True 表示真实观测点
        - t:             [B]         扩散时间步（1 到 T）

    嵌入策略：
        1. 点特征嵌入
           - 输入：[noisy_stec, context_stec] 拼接 → [B, N, 2]
           - 输出：Linear(2 → dim) → [B, N, dim]
           - 作用：提取 STEC 特征

        2. 地理坐标位置编码
           - 输入：coords [B, N, 2]
           - 输出：FourierPosEncoding → [B, N, dim]
           - 作用：编码空间位置信息（lat, lon）

        3. 角度编码
           - 输入：angles [B, N, 2]
           - 输出：AngleFourierEncoding → [B, N, dim]
           - 作用：编码观测几何信息（azimuth, elevation）

        4. 系统 ID 嵌入
           - 输入：system_ids [B, N]
           - 输出：Embedding(10, system_emb_dim) → Linear → [B, N, dim]
           - 作用：区分不同卫星系统（GPS, GLONASS, Galileo, BDS）

        5. 角色类型编码
           - 输入：role_type [B, N]
           - 输出：Embedding(3, dim) → [B, N, dim]
           - 作用：区分 padding / context / target 点

        6. 时间步嵌入（条件信息）
           - 输入：t [B]
           - 输出：TimestepEmbedding → [B, dim]
           - 作用：为 AdaLN 提供时间条件

        7. Context 全局聚合（条件信息）
           - 输入：context_stec [B, N, 1]
           - 输出：均值池化 → MLP → [B, dim]
           - 作用：为 AdaLN 提供空间条件

    Transformer 主干：
        - 6 层 STECBlock（可配置）
        - 每层包含：
          · AdaLN 调制（基于时间步和 context 聚合）
          · Multi-Head Self-Attention（带 RoPE 位置编码）
          · FFN（MLP）
          · Padding mask 支持（忽略 padding 点）

    输出头：
        - RMSNorm + AdaLN 调制
        - Linear(dim → 1)：每个点预测一个标量噪声值
        - 仅 target 位置的输出用于计算 loss

与 JustImageTransformer 的对应关系：
    JiT 组件                    →  STEC 适配
    ─────────────────────────────────────────────────────
    patch_embed                 →  点特征嵌入（Linear）
    t_embedder                  →  TimestepEmbedding（直接迁移）
    y_embedder（类别嵌入）      →  Context 聚合嵌入（从数据聚合）
    JiTBlock（AdaLN）           →  STECBlock（直接迁移并增强）
    final_linear                →  标量输出头（dim → 1）
    RoPE                        →  序列位置编码（保留）
    无                          →  FourierPosEncoding（新增，地理位置）
    无                          →  AngleFourierEncoding（新增，观测角度）
    无                          →  系统 ID 嵌入（新增，多星联合）
    无                          →  角色类型编码（新增，context/target 区分）

设计特点：
    - 多星联合：通过 system_id 嵌入区分不同卫星系统
    - 条件扩散：context 点提供条件信息，target 点进行去噪
    - 空间感知：地理坐标和观测角度的傅里叶编码
    - AdaLN 调制：时间步和 context 聚合作为条件
    - Padding 支持：自动处理变长样本

训练目标：
    给定加噪样本 xt 和条件信息，预测噪声 ε，使得：
        xt = mu_bar(x0, t) + sigma_bar(t) * ε
    模型学习预测 ε，用于反向 SDE 去噪。

参考来源：
    JustImageTransformer（JiT）架构，适配为离散点任务
"""

import torch
import torch.nn as nn
from .blocks import RMSNorm, STECBlock
from .pos_encoding import FourierPosEncoding, AngleFourierEncoding, TimestepEmbedding, precompute_rope_freqs


class STECDiffTransformer(nn.Module):
    """
    基于 Transformer 的离散点集 STEC 噪声预测网络（多星联合版本）。

    Args:
        dim:                模型隐层维度
        depth:              Transformer 层数
        heads:              注意力头数
        mlp_ratio:          FFN 扩展比例
        fourier_bands:      地理坐标傅里叶编码频率数
        angle_fourier_bands: 角度傅里叶编码频率数
        system_emb_dim:     系统ID嵌入维度
        time_emb_dim:       时间步嵌入中间维度
        max_seq_len:        预计算 RoPE 的最大序列长度（需大于单批次最大 IPP 点数）
        dropout:            dropout 概率
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        fourier_bands: int = 64,
        angle_fourier_bands: int = 32,
        system_emb_dim: int = 32,
        time_emb_dim: int = 256,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim   = dim
        self.heads = heads
        head_dim   = dim // heads

        # ------------------------------------------------------------------
        # 1. 输入嵌入层
        # ------------------------------------------------------------------

        # 点特征嵌入：将 [noisy_stec(1), context_stec(1), prior_mu(1), prior_unc(1), prior_gap(1)] 拼接后映射到 dim
        # 即：每个点有 5 个通道作为输入特征
        # 如果不提供 prior_features，则后 3 维用零填充（保持向后兼容）
        self.point_embed = nn.Sequential(
            nn.Linear(5, dim, bias=True),   # 5 = noisy_stec + context_stec + prior_mu + prior_unc + prior_gap
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        # 地理坐标傅里叶位置编码
        self.pos_enc = FourierPosEncoding(fourier_bands=fourier_bands, dim=dim)

        # 角度傅里叶编码（azimuth, elevation）
        self.angle_enc = AngleFourierEncoding(fourier_bands=angle_fourier_bands, dim=dim)

        # 系统ID嵌入（GPS=0, GLONASS=1, Galileo=2, BDS=3, ...）
        # 假设最多支持 4 个系统，padding_idx=0 用于 padding 点，1-4 为已知系统
        # Must match data.dataset.map_system_id_to_index: 0=padding, 1-4=known systems,
        self.system_embed = nn.Embedding(5, system_emb_dim, padding_idx=0)
        # 将 system_emb_dim 映射到 dim
        self.system_proj = nn.Linear(system_emb_dim, dim, bias=True)

        # 角色类型编码（0=padding, 1=context, 2=target）
        # 3 种类型，学习可训练的嵌入向量
        self.role_embed = nn.Embedding(3, dim, padding_idx=0)

        # ------------------------------------------------------------------
        # 2. 条件嵌入（用于 AdaLN）
        # ------------------------------------------------------------------

        # 时间步嵌入
        self.time_embed = TimestepEmbedding(dim=dim, sinusoidal_dim=time_emb_dim)

        # Context 全局聚合 MLP：将 context 特征均值池化后映射到 dim
        # 用于给 AdaLN 提供空间条件信息
        self.context_agg_mlp = nn.Sequential(
            nn.Linear(1, dim, bias=True),  # context 均值 STEC（1维）
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        # ------------------------------------------------------------------
        # 3. Transformer 主干
        # ------------------------------------------------------------------

        self.blocks = nn.ModuleList([
            STECBlock(dim=dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        # ------------------------------------------------------------------
        # 4. 输出头
        # ------------------------------------------------------------------

        self.final_norm  = RMSNorm(dim)
        # AdaLN 最终层
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        # 标量噪声预测头（每个点输出 1 个噪声预测值）
        self.output_head = nn.Linear(dim, 1, bias=True)

        # ------------------------------------------------------------------
        # 5. 预计算 RoPE 频率（注册为 buffer）
        # ------------------------------------------------------------------

        freqs_cis = precompute_rope_freqs(head_dim, max_seq_len=max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化线性层权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        noisy_stec: torch.Tensor,
        coords: torch.Tensor,
        angles: torch.Tensor,
        system_ids: torch.Tensor,
        context_stec: torch.Tensor,
        role_type: torch.Tensor,
        valid_mask: torch.Tensor,
        t: torch.Tensor,
        prior_features: torch.Tensor = None,
        weak_condition: bool = False,
        context_dropout_rate: float = 0.3,
    ) -> torch.Tensor:
        """
        前向传播：预测每个点的噪声 ε。

        Args:
            noisy_stec:     [B, N, 1]   加噪后 STEC（target处是xt，context处是真实值）
            coords:         [B, N, 2]   归一化坐标 (lat_norm, lon_norm)
            angles:         [B, N, 2]   归一化角度 (azimuth_norm, elevation_norm)
            system_ids:     [B, N]      int64，系统ID（0=padding, 1=GPS, 2=GLONASS, 3=BDS）
            context_stec:   [B, N, 1]   context 处真实 STEC，target 和 padding 处为 0
            role_type:      [B, N]      int64，0=padding, 1=context, 2=target
            valid_mask:     [B, N]      bool，True=有效观测点，False=padding
            t:              [B]         扩散时间步（1-indexed）
            prior_features: [B, N, 3]   可选，先验特征 [prior_mu, prior_unc, prior_gap]
                                        如果为 None，则用零填充（保持向后兼容）
            weak_condition: bool        是否使用弱条件分支（第二阶段新增）
            context_dropout_rate: float 弱条件分支的 context dropout 比例（默认 0.3）

        Returns:
            noise_pred: [B, N, 1]   预测的噪声 ε
                        （仅 target 位置有意义，loss 计算时用 target_mask 筛选）
        """
        B, N, _ = noisy_stec.shape

        # ------------------------------------------------------------------
        # Step 0: 弱条件分支 - Context Dropout（第二阶段新增）
        # ------------------------------------------------------------------

        # 如果启用弱条件分支，对 context_stec 进行随机 dropout
        if weak_condition and self.training:
            # 生成 context mask（role_type == 1）
            context_mask = (role_type == 1)  # [B, N]
            # 对每个 context 点以 context_dropout_rate 概率置零
            dropout_mask = torch.rand(B, N, device=context_stec.device) > context_dropout_rate # 此处context_dropout_rate不该称为rate,而是一个阈值
            # 只对 context 点应用 dropout，target 和 padding 点不变
            dropout_mask = dropout_mask | (~context_mask)  # target 和 padding 点保持不变
            # 应用 dropout
            context_stec = context_stec * dropout_mask.unsqueeze(-1).float()

        # ------------------------------------------------------------------
        # Step 1: 构建点特征
        # ------------------------------------------------------------------

        # 如果没有提供 prior_features，用零填充
        if prior_features is None:
            prior_features = torch.zeros(B, N, 3, device=noisy_stec.device, dtype=noisy_stec.dtype)

        # 拼接 noisy_stec, context_stec, prior_features → [B, N, 5]
        point_feat_in = torch.cat([noisy_stec, context_stec, prior_features], dim=-1)
        # 映射到 dim [B, N, dim]
        x = self.point_embed(point_feat_in)

        # ------------------------------------------------------------------
        # Step 2: 加入位置编码、角度编码、系统编码和角色编码
        # ------------------------------------------------------------------

        # 地理坐标傅里叶位置编码 [B, N, dim]
        pos_emb = self.pos_enc(coords)
        x = x + pos_emb

        # 角度傅里叶编码 [B, N, dim]
        angle_emb = self.angle_enc(angles)
        x = x + angle_emb

        # 系统ID嵌入 [B, N, system_emb_dim] → [B, N, dim]
        sys_emb = self.system_embed(system_ids)  # [B, N, system_emb_dim]
        sys_emb = self.system_proj(sys_emb)      # [B, N, dim]
        x = x + sys_emb

        # 角色类型编码 [B, N, dim]
        role_emb = self.role_embed(role_type)  # role_type: [B, N] long
        x = x + role_emb

        # ------------------------------------------------------------------
        # Step 3: 构建条件向量（用于 AdaLN）
        # ------------------------------------------------------------------

        # 时间步嵌入 [B, dim]
        t_emb = self.time_embed(t)

        # Context 全局信息聚合：
        # 对每个样本，取所有 context 点的 STEC 均值作为全局空间条件
        # context_mask: [B, N]  role_type==1 的位置
        context_mask = (role_type == 1)  # [B, N]  bool
        # 计算 context STEC 均值（每个样本独立）[B, 1]
        ctx_sum   = (context_stec.squeeze(-1) * context_mask.float()).sum(dim=1, keepdim=True)
        ctx_count = context_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        ctx_mean  = ctx_sum / ctx_count  # [B, 1]
        # 通过 MLP 映射到 dim [B, dim]
        ctx_emb = self.context_agg_mlp(ctx_mean)

        # 合并时间步和 context 条件 [B, dim]
        cond = t_emb + ctx_emb

        # ------------------------------------------------------------------
        # Step 4: 准备 padding mask 和 RoPE 频率
        # ------------------------------------------------------------------

        # padding mask：True 表示 padding（需要屏蔽）
        pad_mask = ~valid_mask  # [B, N]  bool

        # RoPE 频率：取前 N 个位置 [1, N, 1, head_dim//2]
        head_dim = self.dim // self.heads
        # freqs_cis: [max_seq_len, head_dim//2]  复数
        freqs = self.freqs_cis[:N]  # [N, head_dim//2]
        # reshape 为 [1, N, 1, head_dim//2] 以适配 apply_rope
        freqs = freqs.unsqueeze(0).unsqueeze(2)  # [1, N, 1, head_dim//2]

        # ------------------------------------------------------------------
        # Step 5: Transformer 主干
        # ------------------------------------------------------------------

        for block in self.blocks:
            x = block(x, cond, freqs, pad_mask=pad_mask)  # [B, N, dim]

        # ------------------------------------------------------------------
        # Step 6: 输出头（带 AdaLN 最终层归一化）
        # ------------------------------------------------------------------

        # AdaLN 最终归一化
        shift, scale = self.final_adaLN(cond).chunk(2, dim=1)   # [B, dim]
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 标量噪声预测 [B, N, 1]
        noise_pred = self.output_head(x)

        # padding 位置的输出置零（不影响 loss，但保持张量干净）
        noise_pred = noise_pred * valid_mask.unsqueeze(-1).float()

        return noise_pred


def build_model(cfg: dict) -> STECDiffTransformer:
    """
    根据配置字典构建模型。

    Args:
        cfg: 配置字典（来自 default.yaml 的 model 节）

    Returns:
        STECDiffTransformer 实例
    """
    return STECDiffTransformer(
        dim                 = cfg.get("dim", 256),
        depth               = cfg.get("depth", 6),
        heads               = cfg.get("heads", 8),
        mlp_ratio           = cfg.get("mlp_ratio", 4.0),
        fourier_bands       = cfg.get("fourier_bands", 64),
        angle_fourier_bands = cfg.get("angle_fourier_bands", 32),
        system_emb_dim      = cfg.get("system_emb_dim", 32),
        time_emb_dim        = cfg.get("time_emb_dim", 256),
        max_seq_len         = cfg.get("max_seq_len", 2048),
        dropout             = cfg.get("dropout", 0.0),
    )
