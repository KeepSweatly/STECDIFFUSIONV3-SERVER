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
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    alpha_prev: torch.Tensor,
    target_mask: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    状态域最大似然损失（EDiffSR / IRSDE 思路，论文 Eq.16-17）。

    不在噪声域直接惩罚 ||ε̄ - ε||，而是把它转换到反向状态域：用预测噪声
    走一步反向 SDE 得到的状态估计 x̂_{t-1}，与由真实 x0 推出的理论最优状态
    x*_{t-1} 之间做惩罚。论文指出，反向状态中大部分分量是已知的，
    因此状态域距离比噪声域距离更稳定，能缓解扩散训练的不稳定性。

    本项目 SDE 适配推导：
        反向一步的确定性状态（去掉随机项）为
            x_{t-1} = μ + α_{t-1} * (x0_est - μ)
        其中 x0_est = (x_t - μ - σ_t * ε̄) / α_t + μ。
        代入并令 ε̄ 为预测噪声、ε 为真实噪声，两条状态相减时 μ 与 x_t 抵消，
        闭式差值化简为：
            x̂_{t-1} - x*_{t-1} = (α_{t-1} / α_t) * σ_t * (ε - ε̄)
        即状态域损失等价于「带 (α_{t-1}/α_t)·σ_t 时间权重的噪声残差」。
        t=1 时退化为接近原始噪声残差，t 越大权重随调度变化，
        这正是最大似然加权赋予不同时步的相对重要性。

    为数值稳定，本实现直接用解析权重 coef = (α_{t-1}/α_t) * σ_t 乘到
    噪声残差上（避免显式构造 x̂/x*，也无需 x0_true），与 Eq.16 等价。

    Args:
        noise_pred:   [B, N, 1]   模型预测噪声 ε̄
        noise_target: [B, N, 1]   真实噪声 ε
        alpha_prev:   [B, 1, 1]   每个样本的状态权重系数 (α_{t-1}/α_t) * σ_t
        target_mask:  [B, N]      bool，True 表示 target 点
        loss_type:    "l1" 或 "l2"

    Returns:
        loss: 标量 Tensor
    """
    mask = target_mask.unsqueeze(-1)  # [B, N, 1]

    # 状态域残差 = 系数 * 噪声残差（解析等价于 ||x̂_{t-1} - x*_{t-1}||）
    state_residual = alpha_prev * (noise_pred - noise_target)  # [B, N, 1]

    resid = state_residual[mask]  # [N_target_total]

    if resid.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=noise_pred.device)

    zero = torch.zeros_like(resid)
    if loss_type == "l1":
        return F.l1_loss(resid, zero)
    elif loss_type == "l2":
        return F.mse_loss(resid, zero)
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


def dual_branch_loss(
    noise_pred_strong: torch.Tensor,
    noise_pred_weak: torch.Tensor,
    noise_target: torch.Tensor,
    x0_pred_strong: torch.Tensor,
    x0_true: torch.Tensor,
    noisy_stec_weak: torch.Tensor,
    target_mask: torch.Tensor,
    alpha_prev: torch.Tensor = None,
    lambda_w: float = 0.5,
    lambda_x: float = 0.2,
    lambda_j: float = 1e-4,
    loss_type: str = "l1",
) -> dict:
    """
    双分支条件训练总损失（第二阶段）。

    L_total = L_strong + λ_w * L_weak + λ_x * L_x0 + λ_j * L_jac

    L_strong / L_weak 采用 EDiffSR 的状态域最大似然损失（论文 Eq.16-17）：
    用预测噪声推出的反向状态 x̂_{t-1} 与真实 x0 推出的理论最优状态 x*_{t-1}
    做惩罚，解析等价于「带 (α_{t-1}/α_t)·σ_t 时间权重的噪声残差」。
    若未提供 alpha_prev，则退化为原始噪声域残差（向后兼容）。

    Args:
        noise_pred_strong: [B, N, 1]   强条件分支预测噪声
        noise_pred_weak:   [B, N, 1]   弱条件分支预测噪声
        noise_target:      [B, N, 1]   真实噪声
        x0_pred_strong:    [B, N, 1]   强条件分支恢复的 x0
        x0_true:           [B, N, 1]   真实 x0
        noisy_stec_weak:   [B, N, 1]   弱条件分支输入（requires_grad=True）
        target_mask:       [B, N]      bool，True 表示 target 点
        alpha_prev:        [B, 1, 1]   状态域权重系数 (α_{t-1}/α_t)·σ_t；
                                       None 时退化为噪声域损失
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
    # 1. 强条件分支损失（状态域最大似然，退化时为噪声域）
    if alpha_prev is not None:
        loss_strong = state_maximum_likelihood_loss(
            noise_pred_strong, noise_target, alpha_prev, target_mask, loss_type)
        loss_weak = state_maximum_likelihood_loss(
            noise_pred_weak, noise_target, alpha_prev, target_mask, loss_type)
    else:
        loss_strong = noise_prediction_loss(noise_pred_strong, noise_target, target_mask, loss_type)
        loss_weak = noise_prediction_loss(noise_pred_weak, noise_target, target_mask, loss_type)

    # 3. x0 重建损失（强条件分支）
    loss_x0 = x0_reconstruction_loss(x0_pred_strong, x0_true, target_mask, loss_type)

    # 4. Jacobian 稳定项（弱条件分支噪声对 x_t 的梯度范数）
    loss_jac = jacobian_regularization(noise_pred_weak, noisy_stec_weak, target_mask)

    # 5. 总损失
    loss_total = loss_strong + lambda_w * loss_weak + lambda_x * loss_x0 + lambda_j * loss_jac

    return {
        "loss_total": loss_total,
        "loss_strong": loss_strong,
        "loss_weak": loss_weak,
        "loss_x0": loss_x0,
        "loss_jac": loss_jac,
    }
