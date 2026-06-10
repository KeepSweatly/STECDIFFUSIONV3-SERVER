"""
diffusion/sde.py
=================
STEC 条件扩散 SDE（随机微分方程）封装

概述：
    本模块实现基于 Ornstein-Uhlenbeck 过程的条件图像恢复 SDE，适配于 STEC
    离散点集的条件扩散建模。核心思路来源于 EDiffSR 的 IRSDE 类，针对电离层
    STEC 预测任务进行了适配。

数学基础：
    Ornstein-Uhlenbeck SDE（OU 过程）：
        前向过程：dx = θ(μ - x)dt + σ(t)dW

    条件均值轨迹：
        mu_bar(x0, t) = μ + (x0 - μ) * exp(-θ * t)

    条件噪声标准差（cosine 调度）：
        sigma_bar(t) = max_sigma * (1 - cos(π * t/T)) / 2

    加噪后的 xt：
        xt = mu_bar(x0, t) + sigma_bar(t) * ε,   ε ~ N(0, I)

任务适配说明：
    EDiffSR 原始设计：
        - μ 是上采样后的 LR 图像（规则网格，每像素都有对应 μ 值）
        - 适用于图像超分辨率任务

    本任务适配：
        - μ 是离散点集上的条件均值：
          · context 点：μ = context 点 STEC 的全局均值（标量广播）
          · target 点：μ = IDW 距离加权插值结果（每个 target 点独立计算）
        - 适用于电离层 STEC 空间插值任务

    保留部分：
        - SDE 数学结构（OU 过程、cosine 调度、前/反向步骤）直接迁移
        - 噪声预测训练目标保持一致

核心组件：
    1. IDW 插值（idw_interpolate）
       - 对每个 target 点，利用最近 k 个 context 点做距离加权插值
       - 得到该 target 点的条件均值 μ_target

    2. STEC_IRSDE 类
       - 前向加噪：forward_sample / forward_sample_batch
       - 反向去噪：reverse_sde_step / reverse_sde
       - 条件均值构建：build_mu_batch
       - 噪声调度：_build_schedule（cosine/linear/constant）

使用场景：
    训练阶段：
        1. 构建条件均值 μ（context 均值 + target IDW）
        2. 随机采样时间步 t
        3. 前向加噪得到 xt
        4. 模型预测噪声 ε
        5. 计算噪声预测 loss

    推理阶段：
        1. 构建条件均值 μ
        2. 初始化 target 点为最大噪声状态 x_T
        3. 迭代 T 步反向 SDE 去噪
        4. 得到最终预测 x0
"""

import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# IDW 距离加权插值工具函数
# ======================================================================

def idw_interpolate(
    target_coords: torch.Tensor,
    context_coords: torch.Tensor,
    context_values: torch.Tensor,
    power: float = 2.0,
    k: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对每个 target 点，利用最近 k 个 context 点做 IDW 插值，
    得到该 target 点的条件均值 μ_target。

    Args:
        target_coords:  [M, 2]   target 点归一化坐标 (lat, lon)
        context_coords: [C, 2]   context 点归一化坐标
        context_values: [C, 1]   context 点 STEC 值
        power:          IDW 幂次（越大越局部）
        k:              取最近 k 个邻居
        eps:            数值稳定小量（防止除零）

    Returns:
        mu_target: [M, 1]   每个 target 点的 IDW 插值结果
    """
    M = target_coords.shape[0]
    C = context_coords.shape[0]

    if C == 0:
        # 没有 context 点时，返回全零
        return torch.zeros(M, 1, device=target_coords.device,
                           dtype=target_coords.dtype)

    # 计算所有 target 与 context 点之间的距离 [M, C]
    diff = target_coords.unsqueeze(1) - context_coords.unsqueeze(0)  # [M, C, 2]
    dist = torch.norm(diff, dim=-1)  # [M, C]

    # 取最近 k 个邻居（k 不超过 C）
    k_eff = min(k, C)
    topk_dist, topk_idx = torch.topk(dist, k=k_eff, dim=-1, largest=False)  # [M, k_eff]

    # 计算 IDW 权重：w_i = 1 / (d_i^power + eps)
    weights = 1.0 / (topk_dist.pow(power) + eps)  # [M, k_eff]
    weights_sum = weights.sum(dim=-1, keepdim=True)  # [M, 1]
    weights_norm = weights / weights_sum             # [M, k_eff]，归一化权重

    # 取对应 context 点的值 [M, k_eff, 1]
    topk_values = context_values[topk_idx]  # [M, k_eff, 1]

    # 加权求和得到插值结果 [M, 1]
    mu_target = (weights_norm.unsqueeze(-1) * topk_values).sum(dim=1)  # [M, 1]
    return mu_target


# ======================================================================
# 主 SDE 类
# ======================================================================

class STEC_IRSDE:
    """
    用于 STEC 离散点的条件图像恢复 SDE（迁移自 EDiffSR IRSDE）。

    参数说明：
        max_sigma (float): 最大噪声标准差，控制扩散幅度
        T (int):           扩散总步数
        schedule (str):    噪声调度方式：'cosine' / 'linear' / 'constant'
        eps (float):       数值稳定小量
        idw_power (float): IDW 插值幂次
        idw_k (int):       IDW 最近邻数量
        theta (float):     OU 过程漂移系数（默认 1.0，控制均值回归速度）

        第三~五阶段新增参数：
        guidance_scale_max (float):  最大 guidance 强度（默认 2.0）
        guidance_beta (float):       空间自适应抑制系数（默认 1.0）
        guidance_schedule (str):     时步调度策略：'sin2' / 'linear' / 'constant'
        weak_context_dropout (float): 弱条件分支 context dropout 比例（默认 0.3）
        prior_unc_a1 (float):        prior_unc 距离项权重
        prior_unc_a2 (float):        prior_unc 邻域标准差项权重
        prior_unc_a3 (float):        prior_unc 凸包外指示项权重
        prior_gap_k2 (int):          prior_gap 的对照 IDW 邻居数
    """

    def __init__(
        self,
        max_sigma: float = 2.0,
        T: int = 100,
        schedule: str = "cosine",
        eps: float = 1e-8,
        idw_power: float = 2.0,
        idw_k: int = 5,
        theta: float = 1.0,
        guidance_scale_max: float = 2.0,
        guidance_beta: float = 1.0,
        guidance_schedule: str = "sin2",
        weak_context_dropout: float = 0.3,
        use_reg: bool = True,
        prior_unc_a1: float = 1.0,
        prior_unc_a2: float = 1.0,
        prior_unc_a3: float = 0.5,
        prior_gap_k2: int = 2,
    ):
        self.max_sigma = max_sigma
        self.T         = T
        self.schedule  = schedule
        self.eps       = eps
        self.idw_power = idw_power
        self.idw_k     = idw_k
        self.theta     = theta

        # 第三~五阶段新增参数
        self.guidance_scale_max = guidance_scale_max
        self.guidance_beta = guidance_beta
        self.guidance_schedule = guidance_schedule
        self.weak_context_dropout = weak_context_dropout

        # REG 矫正开关（True=使用REG Jacobian矫正，False=普通CFG风格）
        self.use_reg = use_reg
        self.prior_unc_a1 = prior_unc_a1
        self.prior_unc_a2 = prior_unc_a2
        self.prior_unc_a3 = prior_unc_a3
        self.prior_gap_k2 = prior_gap_k2

        # 预计算各时间步的 sigma_bar 和 alpha（mu_bar 系数）
        self._build_schedule()

    def _build_schedule(self):
        """
        预计算时间步 t=1..T 对应的：
          sigma_bar[t]: 噪声标准差
          alpha[t]:     均值保留系数 exp(-θ * t/T)

        存储为 numpy 数组，索引 0 对应 t=1，索引 T-1 对应 t=T
        """
        t_arr = np.arange(1, self.T + 1, dtype=np.float64)  # [1, 2, ..., T]
        t_norm = t_arr / self.T  # 归一化到 [0, 1]

        # ---- 噪声标准差调度 ----
        if self.schedule == "cosine":
            # cosine 调度：sigma(t) = max_sigma * (1 - cos(π*t/T)) / 2
            self._sigma_bar = self.max_sigma * (1.0 - np.cos(np.pi * t_norm)) / 2.0
        elif self.schedule == "linear":
            # 线性调度
            self._sigma_bar = self.max_sigma * t_norm
        elif self.schedule == "constant":
            self._sigma_bar = np.full_like(t_norm, self.max_sigma)
        else:
            raise ValueError(f"未知的噪声调度方式：{self.schedule}")

        # ---- 均值保留系数 ----
        # alpha(t) = exp(-theta * t/T)，t 越大，x0 的贡献越小
        self._alpha = np.exp(-self.theta * t_norm)  # [T]

    def sigma_bar(self, t: int) -> float:
        """返回时间步 t（1-indexed）对应的噪声标准差"""
        return float(self._sigma_bar[t - 1])

    def alpha(self, t: int) -> float:
        """返回时间步 t 对应的均值保留系数"""
        return float(self._alpha[t - 1])

    def mu_bar(self, x0: torch.Tensor, mu: torch.Tensor, t: int) -> torch.Tensor:
        """
        计算前向过程条件均值轨迹：
            mu_bar(x0, t) = μ + alpha(t) * (x0 - μ)

        Args:
            x0:  [*, 1]   原始 STEC 值
            mu:  [*, 1]   条件均值（context均值或IDW插值）
            t:   时间步（1-indexed）

        Returns:
            mean: [*, 1]
        """
        a = self.alpha(t)
        return mu + a * (x0 - mu) # 难道这里没有定义加噪时的均值系数λ吗？

    # ------------------------------------------------------------------
    # 状态域最大似然损失支撑：反向后验均值（迁移自 EDiffSR reverse_optimum_step）
    # ------------------------------------------------------------------

    def _sigma_bar_tensor(self, t_float: torch.Tensor) -> torch.Tensor:
        """
        解析计算 sigma_bar(t)，支持张量输入且允许 t=0（用于 t-1 时的 σ̄_{t-1}）。

        与 _build_schedule 中的离散调度保持一致：
          cosine:   max_sigma * (1 - cos(π·t/T)) / 2
          linear:   max_sigma * t/T
          constant: max_sigma

        Args:
            t_float: 任意形状的浮点张量（时间步，可含 0）
        Returns:
            sigma_bar: 同形状张量
        """
        t_norm = t_float / self.T
        if self.schedule == "cosine":
            return self.max_sigma * (1.0 - torch.cos(np.pi * t_norm)) / 2.0
        elif self.schedule == "linear":
            return self.max_sigma * t_norm
        elif self.schedule == "constant":
            return torch.full_like(t_float, self.max_sigma)
        else:
            raise ValueError(f"未知的噪声调度方式：{self.schedule}")

    def posterior_coeffs_batch(self, t_batch: torch.Tensor) -> tuple:
        """
        批量计算反向后验 q(x_{t-1} | x_t, x0) 的状态域系数 (c1, c2)：

            x*_{t-1} = μ + c1·(x_t - μ) + c2·(x0 - μ)

        其中（s_t 为单步均值系数，β̃_t² 为单步方差）：
            s_t   = α_t / α_{t-1}
            β̃_t² = σ̄_t² - s_t²·σ̄_{t-1}²
            c1    = s_t · σ̄_{t-1}² / σ̄_t²
            c2    = α_{t-1} · β̃_t² / σ̄_t²

        t=1 时 σ̄_0=0 → c1=0, c2=1，x*_0 自然退化为 x0（无需特判）。

        Args:
            t_batch: [B]  时间步（1-indexed，long）
        Returns:
            c1, c2: 均为 [B, 1, 1]
        """
        t = t_batch.float()
        a_t    = torch.exp(-self.theta * t / self.T)
        a_prev = torch.exp(-self.theta * (t - 1.0) / self.T)
        sig_t    = self._sigma_bar_tensor(t)
        sig_prev = self._sigma_bar_tensor(t - 1.0)

        s_t   = a_t / (a_prev + self.eps)
        beta2 = (sig_t ** 2 - s_t ** 2 * sig_prev ** 2).clamp(min=0.0)
        # σ̄_t² 在 t≥1 恒 >0（cosine 调度下 t=1 时约 2.4e-7），用极小 floor 防除零，
        # 避免 self.eps(1e-8) 在小 t 处显著扭曲系数（否则 t=1 时 c2≈0.96 而非 1.0）
        denom = sig_t ** 2 + 1e-12

        c1 = s_t * sig_prev ** 2 / denom
        c2 = a_prev * beta2 / denom
        return c1.view(-1, 1, 1), c2.view(-1, 1, 1)

    def predict_x0_from_noise(
        self,
        xt: torch.Tensor,
        noise_pred: torch.Tensor,
        mu: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        从加噪样本 xt 和预测噪声恢复 x0 估计（批量）：
            x0_pred = (xt - μ - σ̄_t·ε̄) / α_t + μ

        Args:
            xt:         [B, N, 1]  加噪样本
            noise_pred: [B, N, 1]  预测噪声 ε̄
            mu:         [B, N, 1]  条件均值
            t_batch:    [B]        时间步（1-indexed）
        Returns:
            x0_pred: [B, N, 1]
        """
        t = t_batch.float()
        a_t   = torch.exp(-self.theta * t / self.T).view(-1, 1, 1)
        sig_t = self._sigma_bar_tensor(t).view(-1, 1, 1)
        return (xt - mu - sig_t * noise_pred) / (a_t + self.eps) + mu

    def reverse_posterior_mean(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        mu: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        反向后验均值（批量，状态域），EDiffSR reverse_optimum_step 的项目 SDE 适配版：
            x_{t-1} = μ + c1·(xt - μ) + c2·(x0 - μ)

        - 传入真实 x0 → 得到理论最优状态 x*_{t-1}
        - 传入 x0_pred → 得到预测反向状态 x̂_{t-1}

        Args:
            xt:      [B, N, 1]  加噪样本
            x0:      [B, N, 1]  x0（真实或预测）
            mu:      [B, N, 1]  条件均值
            t_batch: [B]        时间步（1-indexed）
        Returns:
            x_{t-1}: [B, N, 1]
        """
        c1, c2 = self.posterior_coeffs_batch(t_batch)
        return mu + c1 * (xt - mu) + c2 * (x0 - mu)

    def guidance_timestep_schedule(self, t: int) -> float:
        """
        时步自适应 guidance 调度函数（第四阶段）。

        计算时间步 t 对应的调度系数 s(t)，用于调制 guidance 强度。

        支持的调度策略：
            - 'sin2':     s(t) = sin²(π * t / (2*T))  （早期弱，后期强）
            - 'linear':   s(t) = t / T                （线性增长）
            - 'constant': s(t) = 1.0                  （恒定强度）

        Args:
            t: 时间步（1-indexed，1 到 T）

        Returns:
            s_t: 调度系数（0 到 1 之间）
        """
        t_norm = t / self.T  # 归一化到 [0, 1]

        if self.guidance_schedule == "sin2":
            # sin²(π*t/2T)：早期弱（接近0），后期强（接近1）
            import math
            s_t = math.sin(math.pi * t_norm / 2.0) ** 2
        elif self.guidance_schedule == "linear":
            # 线性增长
            s_t = t_norm
        elif self.guidance_schedule == "constant":
            # 恒定强度
            s_t = 1.0
        else:
            raise ValueError(f"未知的 guidance_schedule: {self.guidance_schedule}")

        return s_t

    def compute_spatial_adaptive_weights(
        self,
        prior_unc: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        计算空间自适应 guidance 权重（第五阶段）。

        公式：w_{t,j} = w_max * s(t) * exp(-β * u_j)

        其中：
            - w_max: 最大 guidance 强度（self.guidance_scale_max）
            - s(t):  时步调度系数（guidance_timestep_schedule）
            - β:     不确定度抑制系数（self.guidance_beta）
            - u_j:   点 j 的先验不确定度（prior_unc）

        Args:
            prior_unc: [B, N, 1]  先验不确定度（已归一化到 [0, 1]）
            t:         时间步（1-indexed）

        Returns:
            weights: [B, N, 1]  空间自适应权重
        """
        # 时步调度系数
        s_t = self.guidance_timestep_schedule(t)

        # 空间自适应调制：exp(-β * u_j)
        # 高不确定度区域（u_j 接近 1）→ 权重降低
        # 低不确定度区域（u_j 接近 0）→ 权重接近 1
        spatial_modulation = torch.exp(-self.guidance_beta * prior_unc)

        # 最终权重
        weights = self.guidance_scale_max * s_t * spatial_modulation

        return weights

    def forward_sample(
        self,
        x0: torch.Tensor,
        mu: torch.Tensor,
        t: int,
    ) -> tuple:
        """
        前向加噪：在时间步 t 对 x0 加噪，得到 xt。

            xt = mu_bar(x0, t) + sigma_bar(t) * ε,   ε ~ N(0, I)

        Args:
            x0:  [*, 1]   原始 STEC（target 点）
            mu:  [*, 1]   条件均值（context均值或IDW插值）
            t:   时间步（1-indexed）

        Returns:
            xt:      [*, 1]   加噪后的 STEC
            noise:   [*, 1]   采样的噪声 ε（用于计算 loss）
            mean_t:  [*, 1]   条件均值轨迹
        """
        mean_t = self.mu_bar(x0, mu, t)
        sigma_t = self.sigma_bar(t)
        noise = torch.randn_like(x0)
        xt = mean_t + sigma_t * noise
        return xt, noise, mean_t

    def forward_sample_batch(
        self,
        x0: torch.Tensor,
        mu: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> tuple:
        """
        批量前向加噪：为 batch 中每个样本随机采样不同时间步。

        Args:
            x0:      [B, N_max, 1]  原始 STEC
            mu:      [B, N_max, 1]  每个点的条件均值
            t_batch: [B]            每个样本对应的时间步（1-indexed）

        Returns:
            xt:     [B, N_max, 1]
            noise:  [B, N_max, 1]
            mean_t: [B, N_max, 1]
        """
        # 向量化：将 sigma/alpha 按样本广播，避免 Python 循环
        sigma_arr = torch.tensor(
            [self._sigma_bar[int(t.item()) - 1] for t in t_batch],
            dtype=x0.dtype, device=x0.device,
        ).view(-1, 1, 1)  # [B, 1, 1]
        alpha_arr = torch.tensor(
            [self._alpha[int(t.item()) - 1] for t in t_batch],
            dtype=x0.dtype, device=x0.device,
        ).view(-1, 1, 1)  # [B, 1, 1]

        mean_t = mu + alpha_arr * (x0 - mu)
        noise  = torch.randn_like(x0)
        xt     = mean_t + sigma_arr * noise
        return xt, noise, mean_t

    def build_mu_batch(
        self,
        coords: torch.Tensor,
        stec: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        return_prior_features: bool = False,
    ) -> tuple:
        """
        构建 batch 中每个点的条件均值 μ 和先验特征（可选）：
          - context 点：μ = 当前样本所有 context 点 STEC 的全局均值（标量广播）
          - target  点：μ = IDW 距离加权插值结果

        当 return_prior_features=True 时，额外计算：
          - prior_mu:  μ_IDW(k=5)，即主先验均值
          - prior_unc: u(p) = a1*d_kNN + a2*Std_kNN + a3*outside_hull
          - prior_gap: Δμ = μ_IDW(k=idw_k) - μ_IDW(k=prior_gap_k2)

        Args:
            coords:       [B, N_max, 2]  归一化坐标
            stec:         [B, N_max, 1]  归一化 STEC（真实值）
            context_mask: [B, N_max]     bool
            target_mask:  [B, N_max]     bool
            return_prior_features: 是否返回先验特征（默认 False，保持向后兼容）

        Returns:
            如果 return_prior_features=False:
                mu: [B, N_max, 1]  各点条件均值（padding 点为 0）
            如果 return_prior_features=True:
                (mu, prior_features):
                    mu: [B, N_max, 1]
                    prior_features: [B, N_max, 3]  包含 [prior_mu, prior_unc, prior_gap]
        """
        B, N_max, _ = stec.shape
        mu = torch.zeros_like(stec)  # [B, N_max, 1]

        if return_prior_features:
            prior_features = torch.zeros(B, N_max, 3, device=stec.device, dtype=stec.dtype)
            # prior_features[:, :, 0] = prior_mu
            # prior_features[:, :, 1] = prior_unc
            # prior_features[:, :, 2] = prior_gap

        for i in range(B): # 对batch内部的单个样本进行处理
            c_mask = context_mask[i]  # [N_max]
            t_mask = target_mask[i]   # [N_max]

            c_stec = stec[i, c_mask, :]       # [C, 1]  context STEC 值
            c_coords = coords[i, c_mask, :]   # [C, 2]  context 坐标
            t_coords = coords[i, t_mask, :]   # [M, 2]  target 坐标

            # context 点：μ = context STEC 全局均值（标量广播）
            if c_stec.shape[0] > 0:
                ctx_mean = c_stec.mean()       # 标量
                mu[i, c_mask, :] = ctx_mean

            # target 点：μ = IDW 插值
            if t_coords.shape[0] > 0 and c_coords.shape[0] > 0:
                mu_target = idw_interpolate(
                    t_coords, c_coords, c_stec,
                    power=self.idw_power, k=self.idw_k,
                )  # [M, 1]
                mu[i, t_mask, :] = mu_target

                # 计算先验特征（仅对 target 点）
                if return_prior_features:
                    M = t_coords.shape[0]
                    C = c_coords.shape[0]

                    # 1. prior_mu: 就是 μ_IDW(k=5)
                    prior_mu_target = mu_target  # [M, 1]

                    # 2. prior_gap: Δμ = μ_IDW(k=idw_k) - μ_IDW(k=prior_gap_k2)
                    mu_target_k2 = idw_interpolate(
                        t_coords, c_coords, c_stec,
                        power=self.idw_power, k=self.prior_gap_k2,
                    )  # [M, 1]
                    prior_gap_target = mu_target - mu_target_k2  # [M, 1]

                    # 3. prior_unc: u(p) = a1*d_kNN + a2*Std_kNN + a3*outside_hull
                    # 计算到最近 k 个 context 点的平均距离
                    diff = t_coords.unsqueeze(1) - c_coords.unsqueeze(0)  # [M, C, 2]
                    dist = torch.norm(diff, dim=-1)  # [M, C]
                    k_eff = min(self.idw_k, C)
                    topk_dist, topk_idx = torch.topk(dist, k=k_eff, dim=-1, largest=False)  # [M, k_eff]
                    d_knn = topk_dist.mean(dim=-1, keepdim=True)  # [M, 1]

                    # 计算邻域 context STEC 的加权标准差
                    topk_values = c_stec[topk_idx]  # [M, k_eff, 1]
                    weights = 1.0 / (topk_dist.pow(self.idw_power) + self.eps)  # [M, k_eff]
                    weights_norm = weights / weights.sum(dim=-1, keepdim=True)  # [M, k_eff]
                    weighted_mean = (weights_norm.unsqueeze(-1) * topk_values).sum(dim=1)  # [M, 1]
                    weighted_var = (weights_norm.unsqueeze(-1) * (topk_values - weighted_mean.unsqueeze(1)).pow(2)).sum(dim=1)  # [M, 1]
                    std_knn = torch.sqrt(weighted_var + self.eps)  # [M, 1]

                    # 判断是否在 context 凸包外（简化版：判断是否所有 context 点都在某一侧）
                    # 这里用一个简化指标：如果最近邻距离 > 某个阈值，认为在凸包外
                    # 更严格的凸包判断需要计算几何，这里简化为距离阈值
                    outside_hull = (d_knn > 0.3).float()  # [M, 1]，阈值 0.3 可调 ,这里是指的是选取的几个最临近点的距离的平均值是否大于阈值

                    # 组合不确定度（权重系数来自配置）
                    prior_unc_target = (
                        self.prior_unc_a1 * d_knn
                        + self.prior_unc_a2 * std_knn
                        + self.prior_unc_a3 * outside_hull
                    )  # [M, 1]

                    # 归一化到 [0, 1]（使用 sigmoid 或 tanh）
                    prior_unc_target = torch.tanh(prior_unc_target)  # [M, 1]

                    # 填充到 prior_features
                    t_indices = torch.where(t_mask)[0]  # [M]
                    prior_features[i, t_indices, 0:1] = prior_mu_target
                    prior_features[i, t_indices, 1:2] = prior_unc_target
                    prior_features[i, t_indices, 2:3] = prior_gap_target

            elif t_coords.shape[0] > 0:
                # 无 context 时 target 的μ用0（退化情况）
                mu[i, t_mask, :] = 0.0
                if return_prior_features:
                    # 无 context 时，先验特征全部为 0
                    pass  # 已经初始化为 0

        if return_prior_features:
            return mu, prior_features
        else:
            return mu

    def noise_state(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        推理阶段：给 target 点添加最大噪声，用作反向扩散的起始状态。
            x_T = μ + max_sigma * ε

        Args:
            x:   [B, N_max, 1]  输入（target 处值会被覆盖，context 处保持不变）
            mu:  [B, N_max, 1]  条件均值

        Returns:
            x_T: [B, N_max, 1]
        """
        noise = torch.randn_like(x)
        return mu + self.max_sigma * noise

    # ------------------------------------------------------------------
    # 反向 SDE 单步（DDPM 风格）
    # ------------------------------------------------------------------

    def reverse_sde_step(
        self,
        xt: torch.Tensor,
        noise_pred: torch.Tensor,
        mu: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        反向 SDE 单步去噪（迁移自 EDiffSR reverse_sde_step）。

        已知时间步 t 的带噪样本 xt 和预测噪声，计算 x_{t-1}。

        DDPM 风格反向步：
            x0_pred = (xt - sigma_bar(t) * noise_pred - (1-alpha(t)) * μ) / alpha(t)
            x_{t-1} = mu_bar(x0_pred, t-1) + sigma_bar(t-1) * noise_t-1

        Args:
            xt:          [*, 1]   当前加噪样本
            noise_pred:  [*, 1]   模型预测的噪声 ε
            mu:          [*, 1]   条件均值
            t:           当前时间步（1-indexed，t > 1）

        Returns:
            x_{t-1}: [*, 1]
        """
        sigma_t = self.sigma_bar(t)
        alpha_t = self.alpha(t)

        # 从 xt 和预测噪声恢复 x0 的估计
        # xt = mu_bar(x0, t) + sigma_t * ε
        #     = mu + alpha_t*(x0-mu) + sigma_t * ε
        # => x0_pred = (xt - mu - sigma_t * noise_pred) / alpha_t + mu
        x0_pred = (xt - mu - sigma_t * noise_pred) / (alpha_t + self.eps) + mu

        if t == 1:
            # 最后一步：直接返回去噪结果
            return x0_pred

        # 计算 t-1 步的条件均值轨迹
        mean_prev = self.mu_bar(x0_pred, mu, t - 1)
        sigma_prev = self.sigma_bar(t - 1)

        # 添加随机噪声（DDPM 风格）
        noise = torch.randn_like(xt)
        return mean_prev + sigma_prev * noise

    def reverse_sde(
        self,
        x_T: torch.Tensor,
        mu: torch.Tensor,
        model,
        coords: torch.Tensor,
        angles: torch.Tensor,
        system_ids: torch.Tensor,
        context_stec: torch.Tensor,
        role_type: torch.Tensor,
        valid_mask: torch.Tensor,
        target_mask: torch.Tensor,
        device: torch.device,
        prior_features: torch.Tensor = None,
        use_guidance: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        完整反向 SDE 采样（推理阶段，第三~五阶段：带 Guidance）。

        从最大噪声状态 x_T 出发，迭代 T 步去噪，
        context 点在每步保持真实 STEC 值不变（条件固定）。

        第三~五阶段新增：双分支 Guidance 机制
            1. 每步同时预测强条件和弱条件分支
            2. 计算 guidance 修正：Δε = ε_strong - ε_weak
            3. 时步自适应调度：s(t) = sin²(πt/2T)
            4. 空间自适应调制：w_{t,j} = w_max * s(t) * exp(-β * u_j)
            5. 应用 guidance：ε_guided = ε_strong + w_{t,j} * Δε

        Args:
            x_T:            [B, N_max, 1]   推理起始噪声状态（target处）
            mu:             [B, N_max, 1]   条件均值
            model:          噪声预测网络（STECDiffTransformer）
            coords:         [B, N_max, 2]   归一化坐标
            angles:         [B, N_max, 2]   归一化角度
            system_ids:     [B, N_max]      系统ID
            context_stec:   [B, N_max, 1]   context 处真实 STEC，target 处为 0
            role_type:      [B, N_max]      0=padding, 1=context, 2=target
            valid_mask:     [B, N_max]      bool
            target_mask:    [B, N_max]      bool
            device:         torch.device
            prior_features: [B, N_max, 3]   可选，先验特征 [prior_mu, prior_unc, prior_gap]
            use_guidance:   是否使用 guidance（默认 True）
            verbose:        是否打印进度

        Returns:
            x0_pred: [B, N_max, 1]  最终去噪结果（仅 target 位置有意义）
        """
        xt = x_T.clone()

        # 提取 prior_unc 用于空间自适应（第五阶段）
        if prior_features is not None and use_guidance:
            prior_unc = prior_features[:, :, 1:2]  # [B, N_max, 1]
        else:
            prior_unc = None

        for t in range(self.T, 0, -1):
            if verbose and t % 20 == 0:
                print(f"  反向采样步 t={t}/{self.T}")

            t_tensor = torch.full((xt.shape[0],), t, dtype=torch.long, device=device)

            with torch.no_grad():
                # 强条件分支预测
                noise_pred_strong = model(
                    noisy_stec=xt,
                    coords=coords,
                    angles=angles,
                    system_ids=system_ids,
                    context_stec=context_stec,
                    role_type=role_type,
                    valid_mask=valid_mask,
                    t=t_tensor,
                    prior_features=prior_features,
                    weak_condition=False,
                )  # [B, N_max, 1]

                # 第三~五阶段：Guidance 机制（REG 矫正版）
                if use_guidance:
                    with torch.enable_grad():
                        # 需要对 xt 启用梯度以计算 REG Jacobian 矫正项
                        xt_for_grad = xt.detach().requires_grad_(True)

                        # 强条件分支（用可求梯度的 xt_for_grad 重新前向）
                        noise_pred_strong_grad = model(
                            noisy_stec=xt_for_grad,
                            coords=coords,
                            angles=angles,
                            system_ids=system_ids,
                            context_stec=context_stec,
                            role_type=role_type,
                            valid_mask=valid_mask,
                            t=t_tensor,
                            prior_features=prior_features,
                            weak_condition=False,
                        )  # [B, N_max, 1]

                        # 弱条件分支预测（同样用 xt_for_grad）
                        noise_pred_weak = model(
                            noisy_stec=xt_for_grad,
                            coords=coords,
                            angles=angles,
                            system_ids=system_ids,
                            context_stec=context_stec,
                            role_type=role_type,
                            valid_mask=valid_mask,
                            t=t_tensor,
                            prior_features=prior_features,
                            weak_condition=True,
                            context_dropout_rate=self.weak_context_dropout,
                        )  # [B, N_max, 1]

                        # 原始 CFG 风格 guidance 方向：Δε = ε_strong - ε_weak
                        delta_noise = noise_pred_strong_grad - noise_pred_weak  # [B, N_max, 1]

                        # 计算空间自适应权重（第四+五阶段）
                        if prior_unc is not None:
                            guidance_weights = self.compute_spatial_adaptive_weights(prior_unc, t)
                        else:
                            s_t = self.guidance_timestep_schedule(t)
                            guidance_weights = self.guidance_scale_max * s_t

                        sigma_t_val = self.sigma_bar(t)

                        if self.use_reg:
                            # REG 矫正（论文 Eq. 21）：
                            #   ε_REG = ε_strong + w * Δε * (1 - σ_t * J)
                            jac_diag = torch.autograd.grad(
                                outputs=noise_pred_strong_grad,
                                inputs=xt_for_grad,
                                grad_outputs=torch.ones_like(noise_pred_strong_grad),
                                create_graph=False,
                                retain_graph=False,
                            )[0]  # [B, N_max, 1]

                            reg_factor = 1.0 - sigma_t_val * jac_diag
                            reg_factor = torch.clamp(reg_factor, min=-5.0, max=5.0)

                            noise_pred = noise_pred_strong + guidance_weights * delta_noise * reg_factor
                        else:
                            noise_pred = noise_pred_strong + guidance_weights * delta_noise

                    noise_pred = noise_pred.detach()
                else:
                    noise_pred = noise_pred_strong

            # 反向 SDE 单步（每个样本独立处理，但这里 batch 化实现）
            sigma_t   = self.sigma_bar(t)
            alpha_t   = self.alpha(t)
            x0_pred   = (xt - mu - sigma_t * noise_pred) / (alpha_t + self.eps) + mu

            if t == 1:
                xt = x0_pred
            else:
                mean_prev  = mu + self.alpha(t - 1) * (x0_pred - mu)
                sigma_prev = self.sigma_bar(t - 1)
                noise      = torch.randn_like(xt)
                xt         = mean_prev + sigma_prev * noise

            # context 点每步保持真实 STEC 值不变（条件固定）
            context_mask_bool = (role_type == 1)  # [B, N_max]
            xt[context_mask_bool] = context_stec[context_mask_bool]

        return xt
