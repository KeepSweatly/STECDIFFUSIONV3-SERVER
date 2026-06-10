"""
scripts/train.py
=================
STEC 条件扩散模型训练入口脚本（多星联合版本）

概述：
    本脚本实现多卫星联合训练框架，采用历元级样本组织方式，支持不同卫星系统
    的 IPP 点在同一批次中联合建模，通过条件扩散模型学习 STEC 空间分布。

使用方式：
    cd D:/Phd/IonoModeling/stec_diffusionv2
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --resume experiments/exp_joint_epoch/checkpoints/ckpt_epoch0020.pth

核心功能：
    1. 数据集构建
       - 扫描 model_stations/ 目录中的历元文件
       - 每个历元文件作为一个样本（包含多颗卫星的 IPP 点）
       - 按 80/20 比例划分训练集/验证集（历元级别）
       - 统一归一化：坐标、角度、STEC 值共享归一化参数

    2. 模型训练
       - 多星联合：所有卫星系统的 IPP 点在同一批次中训练
       - IPP 点级划分：每个样本内部随机抽取 target 点（10%-50%）
       - 条件扩散：context 点提供条件信息，target 点进行去噪训练
       - 噪声预测：模型学习预测加噪过程中的噪声分量

    3. 训练策略
       - 优化器：AdamW（权重衰减正则化）
       - 学习率调度：Warmup + Cosine Annealing
       - 梯度裁剪：防止梯度爆炸
       - Early Stopping：基于验证集 MAE，patience=10

    4. 输出管理
       - 归一化参数：保存至 normalizer.json（供推理复用）
       - Checkpoint：定期保存 + 最佳模型保存
       - 日志记录：训练/验证指标实时记录

设计原则：
    - 多星联合：不同卫星系统共享模型参数，提升泛化能力
    - 历元级样本：保持时间一致性，避免时间混淆
    - IPP 点级划分：灵活的 context/target 划分，增强鲁棒性
    - 统一归一化：确保不同卫星系统的数据在同一尺度

训练流程：
    [数据加载] → [模型构建] → [SDE 初始化] → [训练循环] → [验证评估] → [Checkpoint 保存]
"""

import sys
import os
import argparse
import json
import random
import numpy as np
import torch
import yaml
from pathlib import Path

# 将项目根目录加入路径，确保各模块可以正确 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_train_val_datasets
from data.collate import build_dataloader
from models.transformer import build_model
from diffusion.sde import STEC_IRSDE
from training.trainer import Trainer


def set_seed(seed: int):
    """固定全局随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    # ------------------------------------------------------------------
    # 解析命令行参数
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="STEC 条件扩散模型训练脚本（多星联合版）")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "configs", "default.yaml"),
        help="配置文件路径（默认：configs/default.yaml）"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从 checkpoint 恢复训练，传入 .pth 文件路径"
    )
    parser.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="resume 时不加载 checkpoint 中的 scheduler 状态，改用当前配置重新初始化的 scheduler",
    )
    parser.add_argument(
        "--reset-global-step",
        action="store_true",
        help="resume 时将 global_step 重置为 0，使 validation 步数从本次续训重新计数",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="训练完成后不自动运行 test.py 推理"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 加载配置
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    print(f"[Config] 已加载配置：{args.config}")

    # ------------------------------------------------------------------
    # 设置随机种子
    # ------------------------------------------------------------------
    seed = cfg["experiment"].get("seed", 2026)
    set_seed(seed)
    print(f"[Seed] 随机种子：{seed}")

    # ------------------------------------------------------------------
    # 设备
    # ------------------------------------------------------------------
    device_str = cfg["training"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA 不可用，改用 CPU 训练")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[Device] 使用设备：{device}")

    # ------------------------------------------------------------------
    # 1. 构建训练集和验证集
    # ------------------------------------------------------------------
    print("\n[1/6] 构建数据集...")

    # 项目根目录（脚本在 scripts/ 下，根目录在上一级）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_stations_dir = os.path.join(project_root, cfg["data"]["model_stations_dir"])

    if not os.path.exists(model_stations_dir):
        print(f"[Error] 建模数据目录不存在：{model_stations_dir}")
        return

    # 系统过滤配置
    system_filter = cfg["data"].get("system_filter", None)
    if system_filter:
        print(f"  卫星系统过滤：{system_filter}（仅训练该系统的 IPP 数据）")
    else:
        print(f"  卫星系统过滤：无（全系统联合训练）")

    epoch_files = sorted(Path(model_stations_dir).glob("*.csv"))
    if not epoch_files:
        print(f"[Error] 目录 {model_stations_dir} 中未找到任何 CSV 文件")
        return

    print(f"  发现 {len(epoch_files)} 个历元文件")

    train_ds, val_ds, coord_norm, stec_norm, angle_norm = build_train_val_datasets(
        model_stations_dir=model_stations_dir,
        cfg=cfg["data"],
        cache_dir=os.path.join(project_root, ".dataset_cache"),
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = build_dataloader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  训练集：{len(train_ds)} 个历元，{len(train_loader)} 个 batch")
    print(f"  验证集：{len(val_ds)} 个历元，{len(val_loader)} 个 batch")

    # ------------------------------------------------------------------
    # 2. 构建模型
    # ------------------------------------------------------------------
    print("\n[2/6] 构建模型...")
    model_cfg = dict(cfg["model"])
    model = build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量：{n_params:,}")

    # ------------------------------------------------------------------
    # 3. 构建 SDE
    # ------------------------------------------------------------------
    print("\n[3/6] 构建 SDE...")
    sde_cfg = cfg["sde"]
    mu_reg_cfg = cfg.get("mu_reg", {})
    sde = STEC_IRSDE(
        max_sigma=sde_cfg.get("max_sigma", 2.0),
        T=sde_cfg.get("T", 100),
        schedule=sde_cfg.get("schedule", "cosine"),
        eps=sde_cfg.get("eps", 1e-8),
        idw_power=cfg["inference"].get("idw_power", 2.0),
        idw_k=cfg["inference"].get("idw_k", 5),
        theta=sde_cfg.get("theta", 1.0),
        guidance_scale_max=mu_reg_cfg.get("guidance_scale_max", 2.0),
        guidance_beta=mu_reg_cfg.get("guidance_beta", 1.0),
        guidance_schedule=mu_reg_cfg.get("guidance_schedule", "sin2"),
        weak_context_dropout=mu_reg_cfg.get("weak_context_dropout", 0.3),
        use_reg=mu_reg_cfg.get("use_reg", True),
        prior_unc_a1=mu_reg_cfg.get("prior_unc_a1", 1.0),
        prior_unc_a2=mu_reg_cfg.get("prior_unc_a2", 1.0),
        prior_unc_a3=mu_reg_cfg.get("prior_unc_a3", 0.5),
        prior_gap_k2=mu_reg_cfg.get("prior_gap_k2", 2),
    )
    print(f"  SDE: T={sde.T}, max_sigma={sde_cfg.get('max_sigma', 2.0)}, "
          f"schedule={sde_cfg.get('schedule', 'cosine')}, "
          f"guidance_scale_max={mu_reg_cfg.get('guidance_scale_max', 2.0)}")

    # ------------------------------------------------------------------
    # 4. 保存归一化参数（供推理时复用）
    # ------------------------------------------------------------------
    print("\n[4/6] 保存归一化参数...")

    # resume_lr：resume 时覆盖学习率（不纳入命名）
    resume_lr = cfg["training"].get("resume_lr", None)
    if args.resume and resume_lr is not None:
        cfg["training"]["learning_rate"] = resume_lr
        print(f"  [Resume] 学习率已覆盖为 resume_lr={resume_lr}")

    # 构建 setting_tag（关键超参数编码）
    _m   = cfg["model"]
    _sde = cfg["sde"]
    _mu  = cfg.get("mu_reg", {})
    _tr  = cfg["training"]
    _inf = cfg["inference"]
    _dat = cfg["data"]
    sys_tag = f"_{system_filter}" if system_filter else ""
    setting_tag = (
        f"chp_joint{sys_tag}"
        f"_d{_m['dim']}_L{_m['depth']}"
        f"_h{_m['heads']}_mr{_m['mlp_ratio']}"
        f"_sig{_sde['max_sigma']}_g{_mu.get('guidance_scale_max', 2.0)}"
        f"_lr{_tr['learning_rate']}_bs{_tr['batch_size']}"
        f"_ik{_inf['idw_k']}_ip{_inf['idw_power']}"
        f"_mmi{_dat['mask_ratio_min']}_mma{_dat['mask_ratio_max']}"
        f"_wcd{_mu.get('weak_context_dropout', 0.3)}"
        f"_las{_mu.get('lambda_smooth', 1e-3)}"
    )

    out_dir = cfg["experiment"]["output_dir"]
    exp_dir = os.path.join(project_root, out_dir, setting_tag)
    os.makedirs(exp_dir, exist_ok=True)

    norm_path = os.path.join(exp_dir, "normalizer.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump({
            "coord":  coord_norm.state_dict(),
            "stec":   stec_norm.state_dict(),
            "angle":  angle_norm.state_dict(),
        }, f, indent=2, ensure_ascii=False)
    print(f"  归一化参数已保存：{norm_path}")

    # ------------------------------------------------------------------
    # 5. 构建训练器
    # ------------------------------------------------------------------
    print("\n[5/6] 构建训练器...")
    # 将 setting_tag 写回 cfg，确保 Trainer 使用正确的输出目录
    cfg["experiment"]["name"] = setting_tag
    trainer = Trainer(
        model=model,
        sde=sde,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        stec_normalizer=stec_norm,
        device=device,
    )

    # 若指定 resume，从 checkpoint 恢复
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[Error] checkpoint 文件不存在：{args.resume}")
            return
        trainer.load_checkpoint(
            args.resume,
            reset_scheduler=args.reset_scheduler,
            reset_global_step=args.reset_global_step,
        )
        print(f"  已从 checkpoint 恢复：{args.resume}")
        if args.reset_scheduler:
            print("  Scheduler 状态已重置：使用当前配置重新调度学习率")
        if args.reset_global_step:
            print("  global_step 已重置为 0：validation 步数从本次续训重新计数")

    # ------------------------------------------------------------------
    # 6. 启动训练
    # ------------------------------------------------------------------
    print(f"\n[6/6] 启动训练...")
    print(f"  实验标识：{setting_tag}")
    print(f"  输出目录：{exp_dir}")
    print(f"  训练轮数：{cfg['training']['num_epochs']}")
    accum_steps = cfg['training'].get('gradient_accumulation_steps', 1)
    print(f"  批大小：{_tr['batch_size']}（梯度累积 {accum_steps} 步，等效 batch={_tr['batch_size'] * accum_steps}）")
    print(f"  学习率：{cfg['training']['learning_rate']}")
    print()

    trainer.train()

    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)

    # ------------------------------------------------------------------
    # 6.5 释放训练占用的 GPU 显存，避免推理阶段显存不足
    # ------------------------------------------------------------------
    del trainer
    del model
    del sde
    del train_loader
    del val_loader
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"[GPU] 训练资源已释放，当前可用显存：{free_gb:.1f} GB")

    # ------------------------------------------------------------------
    # 7. 训练完成后自动运行推理测试
    # ------------------------------------------------------------------
    if not args.no_test:
        print("\n[7/7] 训练完成，自动启动推理测试...")
        import importlib.util
        test_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.py")
        spec = importlib.util.spec_from_file_location("test_module", test_script)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)

        # 构造 test.py 所需的 sys.argv（传入相同的 config）
        import sys as _sys
        _orig_argv = _sys.argv[:]
        _sys.argv = ["test.py", "--config", args.config]
        try:
            test_module.main()
        finally:
            _sys.argv = _orig_argv
    else:
        print("\n[跳过] --no-test 已指定，不运行推理测试")


if __name__ == "__main__":
    main()
