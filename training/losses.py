"""
training/losses.py
===================
训练损失函数（第二阶段：双分支条件训练）。

设计要点：
  - 只对 target 位置的点计算 loss（context 和 padding 点不参与）
  - 支持 L1 loss（EDiffSR 默认，对异常值更鲁棒）和 L2 loss
  - 第二阶段新增：弱条件损失、x0 重建损失、Jacobian 稳定项
  - 返回标量 loss，方便直接调用 .backward()

第二阶段损失函数：
  L_total = L_strong + λ_w * L_weak + λ_x * L_x0 + λ_j * L_jac

  其中：
    - L_strong: 强条件分支状态域最大似然损失（EDiffSR Eq.16-17，完整 context）
    - L_weak:   弱条件分支状态域最大似然损失（30% context dropout）
    - L_x0:     x0 重建损失（从 xt 和预测噪声恢复 x0）
    - L_jac:    Jacobian 稳定项（弱条件分支噪声对 x_t 的梯度 L2 范数）

  状态域损失（EDiffSR）：不在噪声域惩罚 ||ε̄ - ε||，而是惩罚反向一步的
  状态估计 x̂_{t-1} 与真实 x0 推出的理论最优状态 x*_{t-1} 的距离，
  本项目 SDE 下解析等价于带 (α_{t-1}/α_t)·σ_t 时间权重的噪声残差。
"""

import torch
import torch.nn.functional as F


def noise_prediction_loss(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    target_mask: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    噪声预测损失函数（仅对 target 点计算）。

    Args:
        noise_pred:   [B, N, 1]   模型预测的噪声 ε̂
        noise_target: [B, N, 1]   真实噪声 ε（前向加噪时采样的）
        target_mask:  [B, N]      bool，True 表示 target 点
        loss_type:    "l1" 或 "l2"

    Returns:
        loss: 标量 Tensor
    """
    # 用 target_mask 提取 target 点的预测和真实噪声
    # target_mask: [B, N] → 扩展到 [B, N, 1]
    mask = target_mask.unsqueeze(-1)  # [B, N, 1]

    pred   = noise_pred[mask]    # [N_target_total]
    target = noise_target[mask]  # [N_target_total]

    if target.numel() == 0:
        # 极端情况：没有 target 点，返回 0 loss
        return torch.tensor(0.0, requires_grad=True, device=noise_pred.device)

    if loss_type == "l1":
        return F.l1_loss(pred, target)
    elif loss_type == "l2":
        return F.mse_loss(pred, target)
    else:
        raise ValueError(f"未知的 loss_type: {loss_type}，请选择 'l1' 或 'l2'")


def state_maximum_likelihood_loss(
    xt_1_pred: torch.Tensor,
    xt_1_optimum: torch.Tensor,
    target_mask: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    状态域最大似然损失（EDiffSR / IRSDE 思路，论文 Eq.16-17）。

    严格对齐 EDiffSR denoising_model.optimize_parameters 的实现：
        xt_1_expection = sde.reverse_sde_step_mean(state, score, t)   # x̂_{t-1}
        xt_1_optimum   = sde.reverse_optimum_step(state, state_0, t)  # x*_{t-1}
        loss = loss_fn(xt_1_expection, xt_1_optimum)

    不在噪声域直接惩罚 ||ε̄ - ε||，而是把它转换到反向状态域：
      - x̂_{t-1}：用预测噪声推出 x0_pred，再代入反向后验均值得到的预测状态
      - x*_{t-1}：用真实 x0 代入同一反向后验均值得到的理论最优状态

    关键点（修复前的 bug）：反向后验 x*_{t-1} = μ + c1·(xt-μ) + c2·(x0-μ)
    同时依赖 xt 和 x0（c1 非零），不能退化为仅依赖 x0 的前向均值轨迹
    μ + α_{t-1}·(x0-μ)，否则会丢失后验方差加权，损失退化为常数加权噪声残差。

    本实现接收上层（trainer / sde.reverse_posterior_mean）已构造好的
    x̂_{t-1} 与 x*_{t-1}，仅在 target 点上计算 L1/L2。

    Args:
        xt_1_pred:    [B, N, 1]   预测反向状态 x̂_{t-1}（由 x0_pred 推出）
        xt_1_optimum: [B, N, 1]   理论最优状态 x*_{t-1}（由真实 x0 推出）
        target_mask:  [B, N]      bool，True 表示 target 点
        loss_type:    "l1" 或 "l2"

    Returns:
        loss: 标量 Tensor
    """
    mask = target_mask.unsqueeze(-1)  # [B, N, 1]

    pred = xt_1_pred[mask]      # [N_target_total]
    opt  = xt_1_optimum[mask]   # [N_target_total]

    if pred.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=xt_1_pred.device)

    if loss_type == "l1":
        return F.l1_loss(pred, opt)
    elif loss_type == "l2":
        return F.mse_loss(pred, opt)
    else:
        raise ValueError(f"未知的 loss_type: {loss_type}，请选择 'l1' 或 'l2'")


def x0_reconstruction_loss(
    x0_pred: torch.Tensor,
    x0_true: torch.Tensor,
    target_mask: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    x0 重建损失（第二阶段新增）。

    从 xt 和预测噪声恢复的 x0 与真实 x0 之间的损失。
    公式：x0_pred = (xt - μ - σ_t * ε_pred) / α_t + μ

    Args:
        x0_pred:      [B, N, 1]   从噪声预测恢复的 x0
        x0_true:      [B, N, 1]   真实 x0（原始 STEC）
        target_mask:  [B, N]      bool，True 表示 target 点
        loss_type:    "l1" 或 "l2"

    Returns:
        loss: 标量 Tensor
    """
    mask = target_mask.unsqueeze(-1)  # [B, N, 1]

    pred = x0_pred[mask]
    true = x0_true[mask]

    if true.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=x0_pred.device)

    if loss_type == "l1":
        return F.l1_loss(pred, true)
    elif loss_type == "l2":
        return F.mse_loss(pred, true)
    else:
        raise ValueError(f"未知的 loss_type: {loss_type}")


def jacobian_regularization(
    noise_pred_weak: torch.Tensor,
    noisy_stec: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Jacobian 稳定项（第二阶段新增）。

    计算弱条件分支预测噪声相对于输入 noisy_stec 的梯度 L2 范数。
    目的：防止弱条件分支对输入扰动过于敏感，提升稳定性。

    公式：L_jac = || ∂ε_weak / ∂x_t ||²

    Args:
        noise_pred_weak: [B, N, 1]   弱条件分支预测的噪声（需要 requires_grad=True）
        noisy_stec:      [B, N, 1]   输入的加噪 STEC（需要 requires_grad=True）
        target_mask:     [B, N]      bool，True 表示 target 点

    Returns:
        loss: 标量 Tensor（梯度 L2 范数）
    """
    mask = target_mask.unsqueeze(-1)  # [B, N, 1]

    # 只对 target 点计算 Jacobian
    noise_target = noise_pred_weak[mask]  # [N_target_total]

    if noise_target.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=noise_pred_weak.device)

    # 计算梯度：∂noise_target / ∂noisy_stec
    # 使用 torch.autograd.grad 计算 Jacobian
    # create_graph=True 以支持二阶导数（loss.backward()）
    grad_outputs = torch.ones_like(noise_target)
    grads = torch.autograd.grad(
        outputs=noise_target,
        inputs=noisy_stec,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # [B, N, 1]

    # 只对 target 点的梯度计算 L2 范数
    grads_target = grads[mask]  # [N_target_total]
    jac_loss = torch.mean(grads_target ** 2)

    return jac_loss


def spatial_smoothness_loss(
    x0_pred: torch.Tensor,
    stec_true: torch.Tensor,
    coords: torch.Tensor,
    target_mask: torch.Tensor,
    context_mask: torch.Tensor,
    k: int = 8,
    use_context: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    空间平滑 / 物理正则损失（Dirichlet 能量形式）。

    电离层 STEC 是空间连续缓变场，相邻 IPP 点的 STEC 不应突变。本项对每个
    target 中心点 i，在 target∪context 中取最近 k 个邻居 j，用高斯距离核
    加权惩罚预测值的空间差异：

        L_smooth = mean_i [ Σ_j w_ij·(f_i - val_j)² / Σ_j w_ij ]

    其中：
        f_i   = x0_pred[i]                          （target 中心，参与梯度）
        val_j = x0_pred[j]      若 j 为 target      （参与梯度）
              = stec_true[j].detach() 若 j 为 context（真实观测锚点，当常数）
        w_ij  = exp(-d_ij² / (2·h²))                （高斯核）
        h     = 该样本最近邻距离中位数              （尺度自适应带宽）

    设计要点：
      - 仅作用于强分支预测 x0_pred（物理上有意义的清洁 STEC）
      - 邻居含 context 真实观测做锚定，防止纯 target 自洽塌缩成平面
      - context 邻居 detach，已知条件不被平滑项反向修改
      - 高斯核 + 自适应带宽，只约束很近的点，保留真实空间梯度
      - 权重应设小（建议 1e-3 量级），当弱正则用

    Args:
        x0_pred:      [B, N, 1]  强分支预测的清洁 STEC
        stec_true:    [B, N, 1]  真实 STEC（提供 context 锚值）
        coords:       [B, N, 2]  归一化坐标
        target_mask:  [B, N]     bool，True 表示 target 点
        context_mask: [B, N]     bool，True 表示 context 点
        k:            最近邻数量
        use_context:  邻居是否包含 context 锚点
        eps:          数值稳定小量

    Returns:
        loss: 标量 Tensor
    """
    B = x0_pred.shape[0]
    device = x0_pred.device

    per_point_energy = []  # 收集所有样本所有 target 点的能量

    for i in range(B):
        t_mask = target_mask[i]   # [N]
        c_mask = context_mask[i]  # [N]

        M = int(t_mask.sum().item())
        if M == 0:
            continue

        # 中心：target 点
        center_coords = coords[i, t_mask, :]      # [M, 2]
        center_vals   = x0_pred[i, t_mask, :]      # [M, 1]  参与梯度

        # 候选集：target（预测值，梯度）+ context（真实值，detach）
        cand_coords_list = [center_coords]
        cand_vals_list   = [center_vals]
        if use_context and int(c_mask.sum().item()) > 0:
            cand_coords_list.append(coords[i, c_mask, :])              # [C, 2]
            cand_vals_list.append(stec_true[i, c_mask, :].detach())    # [C, 1] 常数锚点
        cand_coords = torch.cat(cand_coords_list, dim=0)  # [P, 2]
        cand_vals   = torch.cat(cand_vals_list, dim=0)    # [P, 1]
        P = cand_coords.shape[0]

        # 距离矩阵 [M, P]
        diff = center_coords.unsqueeze(1) - cand_coords.unsqueeze(0)  # [M, P, 2]
        dist = torch.norm(diff, dim=-1)                                # [M, P]

        # 排除"自己到自己"（前 M 个候选即 target 本身，对角线置为 +inf）
        eye = torch.eye(M, P, device=device, dtype=torch.bool)
        dist = dist.masked_fill(eye, float("inf"))

        # 取最近 k 个邻居（k 不超过 P-1）
        k_eff = min(k, P - 1)
        if k_eff <= 0:
            continue
        topk_dist, topk_idx = torch.topk(dist, k=k_eff, dim=-1, largest=False)  # [M, k_eff]

        # 自适应带宽 h = 最近邻距离中位数（该样本）
        finite_d = topk_dist[torch.isfinite(topk_dist)]
        if finite_d.numel() == 0:
            continue
        h = finite_d.median().clamp(min=eps)

        # 高斯权重 [M, k_eff]
        w = torch.exp(-topk_dist.pow(2) / (2.0 * h * h + eps))

        # 邻居取值 [M, k_eff, 1]
        neigh_vals = cand_vals[topk_idx]  # [M, k_eff, 1]

        # 加权平方差能量（每个 target 点）[M, 1]
        sq_diff = (center_vals.unsqueeze(1) - neigh_vals).pow(2).squeeze(-1)  # [M, k_eff]
        w_sum = w.sum(dim=-1).clamp(min=eps)                                  # [M]
        energy_i = (w * sq_diff).sum(dim=-1) / w_sum                          # [M]

        per_point_energy.append(energy_i)

    if len(per_point_energy) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    return torch.cat(per_point_energy, dim=0).mean()


def dual_branch_loss(
    noise_pred_strong: torch.Tensor,
    noise_pred_weak: torch.Tensor,
    noise_target: torch.Tensor,
    x0_pred_strong: torch.Tensor,
    x0_true: torch.Tensor,
    noisy_stec_weak: torch.Tensor,
    target_mask: torch.Tensor,
    xt_1_optimum: torch.Tensor = None,
    xt_1_pred_strong: torch.Tensor = None,
    xt_1_pred_weak: torch.Tensor = None,
    coords: torch.Tensor = None,
    context_mask: torch.Tensor = None,
    lambda_w: float = 0.5,
    lambda_x: float = 0.2,
    lambda_j: float = 1e-4,
    lambda_smooth: float = 0.0,
    smooth_k: int = 8,
    smooth_use_context: bool = True,
    loss_type: str = "l1",
) -> dict:
    """
    双分支条件训练总损失（第二阶段）。

    L_total = L_strong + λ_w * L_weak + λ_x * L_x0 + λ_j * L_jac

    L_strong / L_weak 采用 EDiffSR 的状态域最大似然损失（论文 Eq.16-17）：
    惩罚预测反向状态 x̂_{t-1} 与真实 x0 推出的理论最优状态 x*_{t-1} 的距离。
    x̂/x* 由上层（trainer + sde.reverse_posterior_mean）构造后传入：
      - xt_1_optimum:     用真实 x0 推出的 x*_{t-1}（两分支共用）
      - xt_1_pred_strong: 用强分支 x0_pred 推出的 x̂_{t-1}
      - xt_1_pred_weak:   用弱分支 x0_pred 推出的 x̂_{t-1}
    若三者未全部提供，则退化为原始噪声域残差（向后兼容）。

    Args:
        noise_pred_strong: [B, N, 1]   强条件分支预测噪声
        noise_pred_weak:   [B, N, 1]   弱条件分支预测噪声
        noise_target:      [B, N, 1]   真实噪声
        x0_pred_strong:    [B, N, 1]   强条件分支恢复的 x0
        x0_true:           [B, N, 1]   真实 x0
        noisy_stec_weak:   [B, N, 1]   弱条件分支输入（requires_grad=True）
        target_mask:       [B, N]      bool，True 表示 target 点
        xt_1_optimum:      [B, N, 1]   理论最优状态 x*_{t-1}；None 时退化为噪声域
        xt_1_pred_strong:  [B, N, 1]   强分支预测状态 x̂_{t-1}
        xt_1_pred_weak:    [B, N, 1]   弱分支预测状态 x̂_{t-1}
        lambda_w:          弱条件损失权重
        lambda_x:          x0 重建损失权重
        lambda_j:          Jacobian 稳定项权重
        loss_type:         "l1" 或 "l2"

    Returns:
        dict: {
            "loss_total":   总损失（标量）
            "loss_strong":  强条件状态域损失
            "loss_weak":    弱条件状态域损失
            "loss_x0":      x0 重建损失
            "loss_jac":     Jacobian 稳定项
        }
    """
    # 1. 强/弱条件分支损失（状态域最大似然，退化时为噪声域）
    use_state_domain = (
        xt_1_optimum is not None
        and xt_1_pred_strong is not None
        and xt_1_pred_weak is not None
    )
    if use_state_domain:
        loss_strong = state_maximum_likelihood_loss(
            xt_1_pred_strong, xt_1_optimum, target_mask, loss_type)
        loss_weak = state_maximum_likelihood_loss(
            xt_1_pred_weak, xt_1_optimum, target_mask, loss_type)
    else:
        loss_strong = noise_prediction_loss(noise_pred_strong, noise_target, target_mask, loss_type)
        loss_weak = noise_prediction_loss(noise_pred_weak, noise_target, target_mask, loss_type)

    # 3. x0 重建损失（强条件分支）
    loss_x0 = x0_reconstruction_loss(x0_pred_strong, x0_true, target_mask, loss_type)

    # 4. Jacobian 稳定项（弱条件分支噪声对 x_t 的梯度范数）
    loss_jac = jacobian_regularization(noise_pred_weak, noisy_stec_weak, target_mask)

    # 5. 空间平滑 / 物理正则（仅强分支预测 x0，弱正则）
    if coords is not None and context_mask is not None and lambda_smooth > 0:
        loss_smooth = spatial_smoothness_loss(
            x0_pred_strong, x0_true, coords, target_mask, context_mask,
            k=smooth_k, use_context=smooth_use_context)
    else:
        loss_smooth = torch.tensor(0.0, requires_grad=True, device=noise_pred_strong.device)

    # 6. 总损失
    loss_total = (
        loss_strong
        + lambda_w * loss_weak
        + lambda_x * loss_x0
        + lambda_j * loss_jac
        + lambda_smooth * loss_smooth
    )

    return {
        "loss_total": loss_total,
        "loss_strong": loss_strong,
        "loss_weak": loss_weak,
        "loss_x0": loss_x0,
        "loss_jac": loss_jac,
        "loss_smooth": loss_smooth,
    }
