"""
training/trainer.py
====================
STEC 条件扩散模型训练器（多星联合版本）

概述：
    本模块封装完整的训练循环，实现 IPP 点级 context/target 划分策略，
    支持多卫星联合训练、early stopping、checkpoint 管理等功能。

核心功能：
    1. 训练循环管理
       - 初始化模型、优化器、学习率调度器
       - 每个 epoch 遍历训练集
       - 定期在验证集上评估
       - Early stopping 机制（patience=10）

    2. IPP 点级划分策略
       训练阶段：
         - 每个样本包含前 80% IPP 点（训练点池）
         - 从训练点池中随机抽取 10%-50% 作为 target
         - 剩余点作为 context
         - 每个 batch 动态生成 mask，增强鲁棒性

       验证阶段：
         - 每个样本包含所有 IPP 点
         - 前 80% 作为 context（预计算 indices）
         - 后 20% 作为 target（预计算 indices）
         - 固定划分，确保评估一致性

    3. 训练流程（单个 batch）
       a. 生成 context/target mask（训练时随机，验证时固定）
       b. 构建条件均值 μ（context 均值 + target IDW）
       c. 随机采样时间步 t（每个样本独立）
       d. 前向加噪（仅 target 点加噪，context 和 padding 不变）
       e. 模型预测噪声 ε
       f. 计算 loss（仅 target 点）
       g. 反向传播 + 梯度裁剪 + 参数更新

    4. 验证策略
       - 使用中间时间步 T//2 评估去噪能力
       - 单步去噪估计 x0（快速评估，非完整推理）
       - 反归一化后计算 MAE 和 RMSE
       - 基于 MAE 保存最佳模型

    5. 学习率调度
       - Warmup：前 warmup_epochs 轮线性增大学习率
       - Cosine Annealing：warmup 后余弦衰减至 eta_min

    6. Early Stopping
       - 监控验证集 MAE
       - 连续 patience 次无改善则停止训练
       - 防止过拟合，节省计算资源

    7. Checkpoint 管理
       - 定期保存：每 save_every_epochs 轮保存一次
       - 最佳模型：验证 MAE 最低时保存
       - 恢复训练：支持从 checkpoint 恢复训练状态

训练配置参数：
    - num_epochs: 训练轮数
    - batch_size: 批大小
    - learning_rate: 初始学习率
    - weight_decay: 权重衰减系数
    - grad_clip: 梯度裁剪阈值
    - warmup_epochs: warmup 轮数
    - val_start_step: 开始验证的参数更新步数
    - val_interval: 验证间隔（参数更新步数）
    - early_stopping_patience: early stopping 耐心值
    - mask_ratio_min/max: target 点比例范围

输出文件：
    - checkpoints/ckpt_best.pth: 最佳模型
    - checkpoints/ckpt_epoch{N:04d}.pth: 定期保存的模型
    - logs/trainer.log: 训练日志

设计特点：
    - IPP 点级划分：灵活的 context/target 划分，增强泛化能力
    - 动态 mask 生成：训练时每个 batch 随机生成，避免过拟合
    - 快速验证：单步去噪评估，节省时间
    - 自动化管理：early stopping + 最佳模型保存，无需人工干预
"""

import os
import math
import time
import torch
import torch.optim as optim
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
from torch.utils.data import DataLoader

from models.transformer import STECDiffTransformer
from diffusion.sde import STEC_IRSDE
from data.collate import generate_context_target_mask
from training.losses import noise_prediction_loss, dual_branch_loss
from utils.logger import get_logger
from utils.normalizer import STECNormalizer


class Trainer:
    """
    STEC 条件扩散模型训练器(重构版:IPP 点级划分)。

    Args:
        model:           STECDiffTransformer 实例
        sde:             STEC_IRSDE 实例
        train_loader:    训练集 DataLoader
        val_loader:      验证集 DataLoader
        cfg:             完整配置字典(来自 default.yaml)
        stec_normalizer: STECNormalizer(用于反归一化评估指标)
        device:          训练设备
    """

    def __init__(
        self,
        model: STECDiffTransformer,
        sde: STEC_IRSDE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        stec_normalizer: STECNormalizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.sde = sde
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.stec_normalizer = stec_normalizer
        self.device = device

        train_cfg = cfg["training"]
        self.num_epochs = train_cfg["num_epochs"]
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.save_every = train_cfg.get("save_every_epochs", 20)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 10)
        self.loss_type = train_cfg.get("loss_type", "l1")
        self.accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

        # 新增:validation 和 early stopping 配置
        self.val_start_step = train_cfg.get("val_start_step", 1000)
        self.val_interval = train_cfg.get("val_interval", 500)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
        self.val_metric_mode = train_cfg.get("val_metric_mode", "global")

        data_cfg = cfg["data"]
        self.mask_ratio_min = data_cfg.get("mask_ratio_min", 0.10)
        self.mask_ratio_max = data_cfg.get("mask_ratio_max", 0.50)

        # 第二阶段：双分支训练配置
        mu_reg_cfg = cfg.get("mu_reg", {})
        self.lambda_w = mu_reg_cfg.get("lambda_w", 0.5)
        self.lambda_x = mu_reg_cfg.get("lambda_x", 0.2)
        self.lambda_j = mu_reg_cfg.get("lambda_j", 1e-4)
        self.weak_context_dropout = mu_reg_cfg.get("weak_context_dropout", 0.3)

        # 输出目录
        exp_name = cfg["experiment"]["name"]
        out_dir = cfg["experiment"]["output_dir"]
        self.exp_dir = os.path.join(out_dir, exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = get_logger(self.log_dir, name="trainer")

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        # 学习率调度:cosine annealing(warmup 由手动处理)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs - self.warmup_epochs,
            eta_min=1e-6,
        )

        # 训练状态
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_mae = float("inf")
        self.no_improve_count = 0  # early stopping 计数器
        self._epoch_start_time: float = 0.0  # 用于计算训练速度

    # ------------------------------------------------------------------
    # 学习率 warmup
    # ------------------------------------------------------------------

    def _warmup_lr(self, epoch: int):
        """线性 warmup:前 warmup_epochs 轮线性增大学习率"""
        if epoch <= self.warmup_epochs:
            base_lr = self.cfg["training"]["learning_rate"]
            lr = base_lr * epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _after_optimizer_step(self, epoch: int, loss_dict: dict):
        """Bookkeeping that should happen once per real parameter update."""
        self.global_step += 1

        if self.global_step % 100 == 0:
            self.logger.debug(
                f"  [Step {self.global_step}] "
                f"L_total={loss_dict['loss_total'].item():.6f} "
                f"L_strong={loss_dict['loss_strong'].item():.6f} "
                f"L_weak={loss_dict['loss_weak'].item():.6f} "
                f"L_x0={loss_dict['loss_x0'].item():.6f} "
                f"L_jac={loss_dict['loss_jac'].item():.6f}"
            )

        if self.global_step >= self.val_start_step and self.global_step % self.val_interval == 0:
            val_metrics = self._validate()
            val_mae = val_metrics["mae_tecu"]
            val_rmse = val_metrics["rmse_tecu"]
            val_msg = (
                f"  [Val @ step {self.global_step}] "
                f"MAE={val_mae:.4f} TECU  RMSE={val_rmse:.4f} TECU  "
                f"norm_MAE={val_metrics['mae_norm']:.4f}  norm_RMSE={val_metrics['rmse_norm']:.4f}  "
                f"mu_baseline_MAE={val_metrics['mu_mae_tecu']:.4f} TECU  "
                f"mu_baseline_RMSE={val_metrics['mu_rmse_tecu']:.4f} TECU  "
                f"mu_norm_RMSE={val_metrics['mu_rmse_norm']:.4f}"
            )
            self.logger.info(val_msg)
            print(val_msg)

            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.no_improve_count = 0
                self._save_checkpoint(epoch, tag="best")
                improve_msg = f"  => 保存最佳模型(MAE={val_mae:.4f} TECU)"
                self.logger.info(improve_msg)
                print(improve_msg)
            else:
                self.no_improve_count += 1
                no_improve_msg = f"  => 无改善(连续 {self.no_improve_count}/{self.early_stopping_patience})"
                self.logger.info(no_improve_msg)
                print(no_improve_msg)

            self.model.train()

    # ------------------------------------------------------------------
    # 主训练入口
    # ------------------------------------------------------------------

    def train(self):
        """启动完整训练循环"""
        self.logger.info(f"开始训练,共 {self.num_epochs} 轮,设备:{self.device}")
        self.logger.info(f"实验目录:{self.exp_dir}")
        self.logger.info(
            f"Validation 启动参数更新步数:{self.val_start_step},"
            f"间隔:{self.val_interval} 次 optimizer.step"
        )
        self.logger.info(f"Early stopping patience:{self.early_stopping_patience}")
        self.logger.info(f"梯度累积步数:{self.accum_steps},等效 batch_size={self.cfg['training']['batch_size'] * self.accum_steps}")

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # warmup
            if epoch <= self.warmup_epochs:
                self._warmup_lr(epoch)

            epoch_t0 = time.time()

            # 训练一轮
            train_loss = self._train_epoch(epoch)

            epoch_elapsed = time.time() - epoch_t0

            # 余弦调度(warmup 结束后)
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"[Epoch {epoch:03d}/{self.num_epochs}] "
                f"loss={train_loss:.6f}  lr={cur_lr:.2e}  step={self.global_step}"
            )

            # 每 10 个 epoch 打印一次训练速度
            if epoch % 10 == 0 or epoch == self.start_epoch:
                speed_msg = (
                    f"[Speed] Epoch {epoch:03d}/{self.num_epochs}  "
                    f"loss={train_loss:.6f}  lr={cur_lr:.2e}  "
                    f"耗时={epoch_elapsed:.1f}s/epoch  "
                    f"step={self.global_step}"
                )
                print(speed_msg)
                self.logger.info(speed_msg)

            # 定期保存 checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, tag=f"epoch{epoch:04d}")

            # 检查是否触发 early stopping
            if self.no_improve_count >= self.early_stopping_patience:
                self.logger.info(f"Early stopping 触发(连续 {self.early_stopping_patience} 次无改善)")
                break

        self.logger.info("训练完成。")

    # ------------------------------------------------------------------
    # 单轮训练
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """训练一个 epoch（支持梯度累积），返回平均 loss"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # 数据移动到设备
            coords = batch["coords"].to(self.device)           # [B, N, 2]
            angles = batch["angles"].to(self.device)           # [B, N, 2]
            stec = batch["stec"].to(self.device)               # [B, N, 1]
            system_ids = batch["system_ids"].to(self.device)   # [B, N]
            valid_mask = batch["valid_mask"].to(self.device)   # [B, N]

            B, N, _ = stec.shape

            # ---- 1. 生成 context/target mask ----
            context_mask, target_mask = generate_context_target_mask(
                valid_mask.cpu(),
                mask_ratio_min=self.mask_ratio_min,
                mask_ratio_max=self.mask_ratio_max,
                mode="train",
            )
            context_mask = context_mask.to(self.device)
            target_mask = target_mask.to(self.device)

            # ---- 2. 构建角色类型编码 ----
            role_type = torch.zeros(B, N, dtype=torch.long, device=self.device)
            role_type[context_mask] = 1
            role_type[target_mask] = 2

            # ---- 3. 构建 context_stec ----
            context_stec = stec * context_mask.unsqueeze(-1).float()

            # ---- 4. 构建条件均值 μ 和先验特征 ----
            mu, prior_features = self.sde.build_mu_batch(
                coords, stec, context_mask, target_mask,
                return_prior_features=True,
            )

            # ---- 5. 随机采样时间步 t ----
            t_batch = torch.randint(1, self.sde.T + 1, (B,), device=self.device)

            # ---- 6. 前向加噪 ----
            xt_all, noise_all, _ = self.sde.forward_sample_batch(stec, mu, t_batch)
            noisy_stec = stec.clone()
            noisy_stec[target_mask] = xt_all[target_mask]

            # ---- 7. 双分支前向预测 ----
            noise_pred_strong = self.model(
                noisy_stec=noisy_stec, coords=coords, angles=angles,
                system_ids=system_ids, context_stec=context_stec,
                role_type=role_type, valid_mask=valid_mask, t=t_batch,
                prior_features=prior_features, weak_condition=False,
            )

            # 弱条件分支需要 Jacobian（create_graph=True → 二阶导数），
            # Efficient Attention 后端不支持二阶导数，强制走 Math 后端
            noisy_stec_weak = noisy_stec.clone().requires_grad_(True)
            with sdpa_kernel(SDPBackend.MATH):
                noise_pred_weak = self.model(
                    noisy_stec=noisy_stec_weak, coords=coords, angles=angles,
                    system_ids=system_ids, context_stec=context_stec,
                    role_type=role_type, valid_mask=valid_mask, t=t_batch,
                    prior_features=prior_features, weak_condition=True,
                    context_dropout_rate=self.weak_context_dropout,
                )

            sigma_t = torch.tensor([self.sde.sigma_bar(int(t.item())) for t in t_batch], device=self.device)
            alpha_t = torch.tensor([self.sde.alpha(int(t.item())) for t in t_batch], device=self.device)
            sigma_t = sigma_t.view(B, 1, 1)
            alpha_t = alpha_t.view(B, 1, 1)
            x0_pred_strong = (noisy_stec - mu - sigma_t * noise_pred_strong) / (alpha_t + 1e-8) + mu

            # ---- 8. 计算损失（除以累积步数） ----
            loss_dict = dual_branch_loss(
                noise_pred_strong=noise_pred_strong,
                noise_pred_weak=noise_pred_weak,
                noise_target=noise_all,
                x0_pred_strong=x0_pred_strong,
                x0_true=stec,
                noisy_stec_weak=noisy_stec_weak,
                target_mask=target_mask,
                lambda_w=self.lambda_w,
                lambda_x=self.lambda_x,
                lambda_j=self.lambda_j,
                loss_type=self.loss_type,
            )
            loss = loss_dict["loss_total"] / self.accum_steps

            # ---- 9. 反向传播（梯度累积，不立刻更新参数） ----
            loss.backward()

            total_loss += loss_dict["loss_total"].item()
            n_batches += 1

            # ---- 10. 累积够步数后：裁剪梯度 + 更新参数 + 清零梯度 ----
            if (batch_idx + 1) % self.accum_steps == 0:
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._after_optimizer_step(epoch, loss_dict)

        # epoch 结束时若有未消耗的累积梯度,也做一次更新
        if n_batches % self.accum_steps != 0:
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._after_optimizer_step(epoch, loss_dict)

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # 验证(使用预计算的 context/target indices)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_legacy(self) -> tuple:
        """
        在验证集上评估。
        策略:使用数据集预计算的 context_indices(前 80%)和 target_indices(后 20%),
              做一步去噪估计 x0,计算反归一化后的 MAE 和 RMSE。

        支持两种模式（由 self.val_metric_mode 控制）：
          - "global": 汇总所有 target 点后统一计算 MAE/RMSE
          - "per_sample": 每个样本单独计算 RMSE，然后取平均
        """
        self.model.eval()

        if self.val_metric_mode == "per_sample":
            sample_rmse_list = []
            sample_mae_list = []
        else:
            all_pred = []
            all_target = []

        for batch in self.val_loader:
            coords = batch["coords"].to(self.device)           # [B, N, 2]
            angles = batch["angles"].to(self.device)           # [B, N, 2]
            stec = batch["stec"].to(self.device)               # [B, N, 1]
            system_ids = batch["system_ids"].to(self.device)   # [B, N]
            valid_mask = batch["valid_mask"].to(self.device)   # [B, N]
            context_indices_batch = batch.get("context_indices", None)
            target_indices_batch = batch.get("target_indices", None)

            B, N, _ = stec.shape

            context_mask, target_mask = generate_context_target_mask(
                valid_mask.cpu(),
                mode="val",
                context_indices_batch=context_indices_batch,
                target_indices_batch=target_indices_batch,
            )
            context_mask = context_mask.to(self.device)
            target_mask = target_mask.to(self.device)

            if target_mask.sum() == 0:
                continue

            role_type = torch.zeros(B, N, dtype=torch.long, device=self.device)
            role_type[context_mask] = 1
            role_type[target_mask] = 2

            context_stec = stec * context_mask.unsqueeze(-1).float()

            mu, prior_features = self.sde.build_mu_batch(
                coords, stec, context_mask, target_mask,
                return_prior_features=True,
            )

            t_val = self.sde.T // 2
            t_batch = torch.full((B,), t_val, dtype=torch.long, device=self.device)

            xt_all, noise_all, _ = self.sde.forward_sample_batch(stec, mu, t_batch)
            noisy_stec = stec.clone()
            noisy_stec[target_mask] = xt_all[target_mask]

            noise_pred = self.model(
                noisy_stec=noisy_stec,
                coords=coords,
                angles=angles,
                system_ids=system_ids,
                context_stec=context_stec,
                role_type=role_type,
                valid_mask=valid_mask,
                t=t_batch,
                prior_features=prior_features,
                weak_condition=False,
            )

            sigma_t = self.sde.sigma_bar(t_val)
            alpha_t = self.sde.alpha(t_val)

            if self.val_metric_mode == "per_sample":
                for i in range(B):
                    t_mask_i = target_mask[i]
                    if t_mask_i.sum() == 0:
                        continue
                    xt_i = noisy_stec[i, t_mask_i]
                    mu_i = mu[i, t_mask_i]
                    eps_i = noise_pred[i, t_mask_i]
                    x0_pred_i = (xt_i - mu_i - sigma_t * eps_i) / (alpha_t + 1e-8) + mu_i
                    x0_true_i = stec[i, t_mask_i]

                    pred_orig_i = self.stec_normalizer.inverse_transform(x0_pred_i.cpu().numpy().flatten())
                    true_orig_i = self.stec_normalizer.inverse_transform(x0_true_i.cpu().numpy().flatten())

                    diff_i = pred_orig_i - true_orig_i
                    sample_rmse_list.append(float(np.sqrt(np.mean(diff_i ** 2))))
                    sample_mae_list.append(float(np.mean(np.abs(diff_i))))
            else:
                xt_target = noisy_stec[target_mask]
                mu_target = mu[target_mask]
                noise_pred_t = noise_pred[target_mask]
                x0_pred = (xt_target - mu_target - sigma_t * noise_pred_t) / (alpha_t + 1e-8) + mu_target
                x0_true = stec[target_mask]
                all_pred.append(x0_pred.cpu())
                all_target.append(x0_true.cpu())

        if self.val_metric_mode == "per_sample":
            if len(sample_rmse_list) == 0:
                self.logger.warning("验证集中没有有效的 target 点,跳过本次 validation")
                return float("inf"), float("inf")
            mae = float(np.mean(sample_mae_list))
            rmse = float(np.mean(sample_rmse_list))
        else:
            if len(all_pred) == 0:
                self.logger.warning("验证集中没有有效的 target 点,跳过本次 validation")
                return float("inf"), float("inf")
            all_pred = torch.cat(all_pred, dim=0).numpy()
            all_target = torch.cat(all_target, dim=0).numpy()
            pred_orig = self.stec_normalizer.inverse_transform(all_pred.flatten())
            target_orig = self.stec_normalizer.inverse_transform(all_target.flatten())
            mae = float(np.mean(np.abs(pred_orig - target_orig)))
            rmse = float(np.sqrt(np.mean((pred_orig - target_orig) ** 2)))

        return mae, rmse

    def _metric_summary(self, pred_norm: np.ndarray, target_norm: np.ndarray) -> dict:
        """Compute metrics in normalized STEC space and original TECU units."""
        pred_norm = pred_norm.reshape(-1)
        target_norm = target_norm.reshape(-1)

        diff_norm = pred_norm - target_norm
        pred_orig = self.stec_normalizer.inverse_transform(pred_norm)
        target_orig = self.stec_normalizer.inverse_transform(target_norm)
        diff_orig = pred_orig - target_orig

        return {
            "mae_norm": float(np.mean(np.abs(diff_norm))),
            "rmse_norm": float(np.sqrt(np.mean(diff_norm ** 2))),
            "mae_tecu": float(np.mean(np.abs(diff_orig))),
            "rmse_tecu": float(np.sqrt(np.mean(diff_orig ** 2))),
        }

    def _empty_val_metrics(self) -> dict:
        return {
            "mae_tecu": float("inf"),
            "rmse_tecu": float("inf"),
            "mae_norm": float("inf"),
            "rmse_norm": float("inf"),
            "mu_mae_tecu": float("inf"),
            "mu_rmse_tecu": float("inf"),
            "mu_mae_norm": float("inf"),
            "mu_rmse_norm": float("inf"),
        }

    @torch.no_grad()
    def _validate(self) -> dict:
        """
        Validate one-step denoising and report both normalized-space and TECU metrics.

        The primary early-stopping metric remains MAE in TECU. The extra normalized
        metrics and mu/IDW baseline metrics are diagnostics for scale and baseline checks.
        """
        self.model.eval()

        if self.val_metric_mode == "per_sample":
            sample_rmse_list = []
            sample_mae_list = []

        all_pred = []
        all_target = []
        all_mu = []

        for batch in self.val_loader:
            coords = batch["coords"].to(self.device)           # [B, N, 2]
            angles = batch["angles"].to(self.device)           # [B, N, 2]
            stec = batch["stec"].to(self.device)               # [B, N, 1]
            system_ids = batch["system_ids"].to(self.device)   # [B, N]
            valid_mask = batch["valid_mask"].to(self.device)   # [B, N]
            context_indices_batch = batch.get("context_indices", None)
            target_indices_batch = batch.get("target_indices", None)

            B, N, _ = stec.shape

            context_mask, target_mask = generate_context_target_mask(
                valid_mask.cpu(),
                mode="val",
                context_indices_batch=context_indices_batch,
                target_indices_batch=target_indices_batch,
            )
            context_mask = context_mask.to(self.device)
            target_mask = target_mask.to(self.device)

            if target_mask.sum() == 0:
                continue

            role_type = torch.zeros(B, N, dtype=torch.long, device=self.device)
            role_type[context_mask] = 1
            role_type[target_mask] = 2

            context_stec = stec * context_mask.unsqueeze(-1).float()

            mu, prior_features = self.sde.build_mu_batch(
                coords, stec, context_mask, target_mask,
                return_prior_features=True,
            )

            t_val = self.sde.T // 2
            t_batch = torch.full((B,), t_val, dtype=torch.long, device=self.device)

            xt_all, noise_all, _ = self.sde.forward_sample_batch(stec, mu, t_batch)
            noisy_stec = stec.clone()
            noisy_stec[target_mask] = xt_all[target_mask]

            noise_pred = self.model(
                noisy_stec=noisy_stec,
                coords=coords,
                angles=angles,
                system_ids=system_ids,
                context_stec=context_stec,
                role_type=role_type,
                valid_mask=valid_mask,
                t=t_batch,
                prior_features=prior_features,
                weak_condition=False,
            )

            sigma_t = self.sde.sigma_bar(t_val)
            alpha_t = self.sde.alpha(t_val)

            xt_target = noisy_stec[target_mask]
            mu_target = mu[target_mask]
            noise_pred_t = noise_pred[target_mask]
            x0_pred = (xt_target - mu_target - sigma_t * noise_pred_t) / (alpha_t + 1e-8) + mu_target
            x0_true = stec[target_mask]

            all_pred.append(x0_pred.cpu())
            all_target.append(x0_true.cpu())
            all_mu.append(mu_target.cpu())

            if self.val_metric_mode == "per_sample":
                for i in range(B):
                    t_mask_i = target_mask[i]
                    if t_mask_i.sum() == 0:
                        continue

                    xt_i = noisy_stec[i, t_mask_i]
                    mu_i = mu[i, t_mask_i]
                    eps_i = noise_pred[i, t_mask_i]
                    x0_pred_i = (xt_i - mu_i - sigma_t * eps_i) / (alpha_t + 1e-8) + mu_i
                    x0_true_i = stec[i, t_mask_i]
                    stats_i = self._metric_summary(
                        x0_pred_i.cpu().numpy().flatten(),
                        x0_true_i.cpu().numpy().flatten(),
                    )
                    sample_rmse_list.append(stats_i["rmse_tecu"])
                    sample_mae_list.append(stats_i["mae_tecu"])

        if len(all_pred) == 0:
            self.logger.warning("Validation skipped: no valid target points.")
            return self._empty_val_metrics()

        all_pred = torch.cat(all_pred, dim=0).numpy()
        all_target = torch.cat(all_target, dim=0).numpy()
        all_mu = torch.cat(all_mu, dim=0).numpy()

        metrics = self._metric_summary(all_pred, all_target)
        mu_metrics = self._metric_summary(all_mu, all_target)

        if self.val_metric_mode == "per_sample":
            if len(sample_rmse_list) == 0:
                self.logger.warning("Validation skipped: no valid per-sample target points.")
                return self._empty_val_metrics()
            metrics["mae_tecu"] = float(np.mean(sample_mae_list))
            metrics["rmse_tecu"] = float(np.mean(sample_rmse_list))

        metrics.update({
            "mu_mae_tecu": mu_metrics["mae_tecu"],
            "mu_rmse_tecu": mu_metrics["rmse_tecu"],
            "mu_mae_norm": mu_metrics["mae_norm"],
            "mu_rmse_norm": mu_metrics["rmse_norm"],
        })

        return metrics

    # ------------------------------------------------------------------
    # Checkpoint 保存与加载
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, tag: str = ""):
        """保存模型权重和训练状态"""
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{tag}.pth")
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_mae": self.best_val_mae,
            "no_improve_count": self.no_improve_count,
        }, ckpt_path)
        self.logger.debug(f"  checkpoint 已保存:{ckpt_path}")

    def _load_checkpoint_legacy(self, ckpt_path: str):
        """从 checkpoint 恢复训练状态"""
        assert os.path.exists(ckpt_path), f"checkpoint 不存在:{ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_mae = ckpt.get("best_val_mae", float("inf"))
        self.no_improve_count = ckpt.get("no_improve_count", 0)
        self.logger.info(f"从 {ckpt_path} 恢复训练,下一轮从 epoch {self.start_epoch} 开始")

    def load_checkpoint(
        self,
        ckpt_path: str,
        reset_scheduler: bool = False,
        reset_global_step: bool = False,
    ):
        """Resume training from checkpoint, with optional scheduler/step reset."""
        assert os.path.exists(ckpt_path), f"checkpoint 不存在:{ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])

        if reset_scheduler:
            base_lr = self.cfg["training"]["learning_rate"]
            for pg in self.optimizer.param_groups:
                pg["lr"] = base_lr
            self.logger.info("Reset scheduler state; using scheduler initialized from current config.")
        else:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])

        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = 0 if reset_global_step else ckpt.get("global_step", 0)
        self.best_val_mae = ckpt.get("best_val_mae", float("inf"))
        self.no_improve_count = ckpt.get("no_improve_count", 0)

        if reset_global_step:
            self.logger.info("Reset global_step to 0 for the resumed training segment.")

        self.logger.info(f"Resume from {ckpt_path}; next epoch: {self.start_epoch}")
