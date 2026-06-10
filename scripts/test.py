"""
scripts/test.py
================
STEC 条件扩散模型推理/测试入口脚本（多星联合版本）

概述：
    本脚本实现多卫星联合推理，通过历元对齐方式，利用 model_stations 的观测数据
    作为 context，预测 val_stations 的 STEC 值，并按卫星分组导出结果。

使用方式：
    cd D:/Phd/IonoModeling/stec_diffusionv2
    python scripts/test.py
    python scripts/test.py --config configs/default.yaml
    python scripts/test.py --checkpoint experiments/exp_joint_epoch_BDS/checkpoints/ckpt_best.pth
    python scripts/test.py --verbose

核心功能：
    1. 历元文件扫描与对齐
       - 扫描 val_stations/ 目录中的历元文件（测试目标）
       - 匹配 model_stations/ 中同名历元文件（context 来源）
       - 仅处理两个目录中共同存在的历元（文件名匹配）

    2. 多星联合推理
       - model_stations 所有 IPP 点 → context（已知 STEC）
       - val_stations 所有 IPP 点 → target（待预测 STEC）
       - 所有卫星的 IPP 点在同一样本中联合处理

    3. 完整反向 SDE 推理
       - 构建 IDW 条件均值 μ（context 均值 + target IDW 插值）
       - 初始化 target 点为最大噪声状态
       - 迭代 T 步反向 SDE 去噪，恢复 STEC 值

    4. 结果导出与分析
       - 导出完整预测结果（all_predictions.csv）
       - 按 satellite_id + system_id 分组统计（summary_by_satellite.csv）
       - 整体指标 JSON（metrics.json）

设计原则：
    - 多星联合：不同卫星系统的 IPP 点在同一样本中联合建模
    - 历元对齐：确保 context 和 target 来自同一时刻的观测
    - 结果分组：支持按卫星分析不同系统的预测性能
    - 可扩展性：支持任意数量的卫星系统和测站

输出文件：
    - results/final_test/all_predictions.csv: 完整预测结果（包含所有元信息）
    - results/final_test/summary_by_satellite.csv: 按卫星分组的统计摘要
    - results/final_test/metrics.json: 整体评估指标（MAE, RMSE）
"""

import sys
import os
import argparse
import json
import torch
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.sde import STEC_IRSDE
from models.transformer import build_model
from utils.normalizer import CoordNormalizer, STECNormalizer
from data.dataset import map_system_id_to_index, get_system_ascii_code


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_normalizers(norm_path: str):
    """从 JSON 文件加载归一化参数"""
    with open(norm_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    coord_norm = CoordNormalizer()
    coord_norm.load_state_dict(d["coord"])
    stec_norm = STECNormalizer()
    stec_norm.load_state_dict(d["stec"])
    angle_norm = CoordNormalizer()
    if "angle" in d:
        angle_norm.load_state_dict(d["angle"])
    return coord_norm, stec_norm, angle_norm


def _extract_epoch_arrays(model_df: pd.DataFrame, val_df: pd.DataFrame, system_ascii_code):
    """从两个 DataFrame 提取并过滤出单历元所需的 numpy 数组。"""
    if system_ascii_code is not None:
        model_df = model_df[model_df["system_id"] == system_ascii_code].reset_index(drop=True)
        val_df   = val_df[val_df["system_id"]   == system_ascii_code].reset_index(drop=True)

    ctx_lats    = model_df["ipp_latitude"].values.astype(np.float32)
    ctx_lons    = model_df["ipp_longitude"].values.astype(np.float32)
    ctx_az      = model_df["azimuth_deg"].values.astype(np.float32)
    ctx_el      = model_df["elevation_deg"].values.astype(np.float32)
    ctx_stec    = model_df["stec"].values.astype(np.float32)
    ctx_sys_ids = map_system_id_to_index(model_df["system_id"].values.astype(np.int64))
    ctx_sat_ids = model_df["satellite_id"].values.astype(np.int64)
    ctx_stations = model_df["station_name"].tolist()

    tgt_lats    = val_df["ipp_latitude"].values.astype(np.float32)
    tgt_lons    = val_df["ipp_longitude"].values.astype(np.float32)
    tgt_az      = val_df["azimuth_deg"].values.astype(np.float32)
    tgt_el      = val_df["elevation_deg"].values.astype(np.float32)
    tgt_stec    = val_df["stec"].values.astype(np.float32)
    tgt_sys_ids = map_system_id_to_index(val_df["system_id"].values.astype(np.int64))
    tgt_sat_ids = val_df["satellite_id"].values.astype(np.int64)
    tgt_stations = val_df["station_name"].tolist()

    return (ctx_lats, ctx_lons, ctx_az, ctx_el, ctx_stec, ctx_sys_ids, ctx_sat_ids, ctx_stations,
            tgt_lats, tgt_lons, tgt_az, tgt_el, tgt_stec, tgt_sys_ids, tgt_sat_ids, tgt_stations)


def estimate_infer_batch_size(
    valid_epochs: list,
    model_files: dict,
    val_files: dict,
    system_ascii_code,
    device: torch.device,
    model_n_params: int,
    model_dim: int = 256,
    model_depth: int = 3,
    max_batch: int = 32,
) -> int:
    """
    根据 GPU 可用显存自动估算推理 batch size。

    采样前 min(20, N) 个历元的点数，取 P95 作为代表性点数，
    用经验公式估算单样本激活显存，结合可用显存得到 batch size。
    CPU 设备直接返回 1。
    """
    if device.type != "cuda":
        return 1

    # 采样点数
    sample_n = min(20, len(valid_epochs))
    point_counts = []
    for stem in valid_epochs[:sample_n]:
        mdf = pd.read_csv(model_files[stem])
        vdf = pd.read_csv(val_files[stem])
        if system_ascii_code is not None:
            mdf = mdf[mdf["system_id"] == system_ascii_code]
            vdf = vdf[vdf["system_id"] == system_ascii_code]
        point_counts.append(len(mdf) + len(vdf))

    n_max_repr = int(np.percentile(point_counts, 95)) if point_counts else 512

    # 经验公式：激活显存 ≈ N × dim × depth × 4 tensors × 4 bytes × 保守系数3
    bytes_per_sample = n_max_repr * model_dim * model_depth * 4 * 4 * 3
    # 模型参数固定占用（fp32）
    model_bytes = model_n_params * 4

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    # 留出 20% 余量
    usable_bytes = free_bytes * 0.8 - model_bytes
    if usable_bytes <= 0:
        return 1

    batch_size = max(1, min(int(usable_bytes // bytes_per_sample), max_batch))
    print(f"[InferBatch] 代表性点数={n_max_repr}, 可用显存={free_bytes/1e9:.1f}GB, "
          f"估算 infer_batch_size={batch_size}")
    return batch_size


def collate_inference_batch(
    epoch_data_list: list,
    coord_norm: CoordNormalizer,
    stec_norm: STECNormalizer,
    angle_norm: CoordNormalizer,
    device: torch.device,
):
    """
    将多个历元的数据打包成一个 batch tensor，用于批量推理。

    Args:
        epoch_data_list: list of tuples，每个元素为 _extract_epoch_arrays 的返回值
        coord_norm, stec_norm, angle_norm: 归一化器
        device: 目标设备

    Returns:
        batch_tensors: dict，包含 coords/angles/stec_full/sys_ids/valid_mask/
                       context_mask/target_mask/role_type/context_stec，均为 [B, N_max, *]
        meta_list: list of dict，每个历元的元信息（n_ctx, tgt_* 原始数组）
    """
    B = len(epoch_data_list)
    n_totals = []
    normalized_list = []
    meta_list = []

    for data in epoch_data_list:
        (ctx_lats, ctx_lons, ctx_az, ctx_el, ctx_stec,
         ctx_sys_ids, ctx_sat_ids, ctx_stations,
         tgt_lats, tgt_lons, tgt_az, tgt_el, tgt_stec,
         tgt_sys_ids, tgt_sat_ids, tgt_stations) = data

        n_ctx = len(ctx_lats)
        n_tgt = len(tgt_lats)
        n_total = n_ctx + n_tgt

        all_lats    = np.concatenate([ctx_lats, tgt_lats])
        all_lons    = np.concatenate([ctx_lons, tgt_lons])
        all_az      = np.concatenate([ctx_az,   tgt_az])
        all_el      = np.concatenate([ctx_el,   tgt_el])
        all_stec    = np.concatenate([ctx_stec, tgt_stec])
        all_sys_ids = np.concatenate([ctx_sys_ids, tgt_sys_ids])

        coords_n = coord_norm.transform(all_lats, all_lons).astype(np.float32)   # [N, 2]
        angles_n = angle_norm.transform(all_az, all_el).astype(np.float32)       # [N, 2]
        stec_n   = stec_norm.transform(all_stec).astype(np.float32)[:, None]     # [N, 1]

        n_totals.append(n_total)
        normalized_list.append((coords_n, angles_n, stec_n, all_sys_ids, n_ctx))
        meta_list.append({
            "n_ctx": n_ctx, "n_tgt": n_tgt,
            "tgt_lats": tgt_lats, "tgt_lons": tgt_lons,
            "tgt_az": tgt_az, "tgt_el": tgt_el,
            "tgt_stec": tgt_stec,
            "tgt_sys_ids": tgt_sys_ids, "tgt_sat_ids": tgt_sat_ids,
            "tgt_stations": tgt_stations,
        })

    N_max = max(n_totals)

    coords_b    = np.zeros((B, N_max, 2), dtype=np.float32)
    angles_b    = np.zeros((B, N_max, 2), dtype=np.float32)
    stec_b      = np.zeros((B, N_max, 1), dtype=np.float32)
    sys_ids_b   = np.zeros((B, N_max),    dtype=np.int64)
    valid_b     = np.zeros((B, N_max),    dtype=bool)
    ctx_mask_b  = np.zeros((B, N_max),    dtype=bool)
    tgt_mask_b  = np.zeros((B, N_max),    dtype=bool)

    for i, (coords_n, angles_n, stec_n, all_sys_ids, n_ctx) in enumerate(normalized_list):
        n = n_totals[i]
        coords_b[i, :n]   = coords_n
        angles_b[i, :n]   = angles_n
        stec_b[i, :n]     = stec_n
        sys_ids_b[i, :n]  = all_sys_ids
        valid_b[i, :n]    = True
        ctx_mask_b[i, :n_ctx]    = True
        tgt_mask_b[i, n_ctx:n]   = True

    coords_t    = torch.from_numpy(coords_b).to(device)
    angles_t    = torch.from_numpy(angles_b).to(device)
    stec_t      = torch.from_numpy(stec_b).to(device)
    sys_ids_t   = torch.from_numpy(sys_ids_b).to(device)
    valid_t     = torch.from_numpy(valid_b).to(device)
    ctx_mask_t  = torch.from_numpy(ctx_mask_b).to(device)
    tgt_mask_t  = torch.from_numpy(tgt_mask_b).to(device)

    role_type = torch.zeros(B, N_max, dtype=torch.long, device=device)
    role_type[ctx_mask_t] = 1
    role_type[tgt_mask_t] = 2

    ctx_stec_t = stec_t * ctx_mask_t.unsqueeze(-1).float()

    batch_tensors = {
        "coords":        coords_t,
        "angles":        angles_t,
        "stec_full":     stec_t,
        "sys_ids":       sys_ids_t,
        "valid_mask":    valid_t,
        "context_mask":  ctx_mask_t,
        "target_mask":   tgt_mask_t,
        "role_type":     role_type,
        "context_stec":  ctx_stec_t,
    }
    return batch_tensors, meta_list


def predict_batch(
    model,
    sde: STEC_IRSDE,
    batch_tensors: dict,
    meta_list: list,
    stec_norm: STECNormalizer,
    device: torch.device,
    verbose: bool = False,
) -> list:
    """
    对一个 batch 的历元执行批量反向 SDE 推理，返回各历元的 result_df 列表。
    """
    coords       = batch_tensors["coords"]
    angles       = batch_tensors["angles"]
    stec_full    = batch_tensors["stec_full"]
    sys_ids      = batch_tensors["sys_ids"]
    valid_mask   = batch_tensors["valid_mask"]
    context_mask = batch_tensors["context_mask"]
    target_mask  = batch_tensors["target_mask"]
    role_type    = batch_tensors["role_type"]
    context_stec = batch_tensors["context_stec"]

    mu, prior_features = sde.build_mu_batch(
        coords, stec_full, context_mask, target_mask,
        return_prior_features=True,
    )

    # 初始化噪声起始状态（仅 target 点）
    x_T = stec_full.clone()
    target_noise = mu + sde.max_sigma * torch.randn_like(stec_full)
    x_T[target_mask] = target_noise[target_mask]

    x0_pred = sde.reverse_sde(
        x_T=x_T,
        mu=mu,
        model=model,
        coords=coords,
        angles=angles,
        system_ids=sys_ids,
        context_stec=context_stec,
        role_type=role_type,
        valid_mask=valid_mask,
        target_mask=target_mask,
        device=device,
        prior_features=prior_features,
        verbose=verbose,
    )  # [B, N_max, 1]

    result_dfs = []
    for i, meta in enumerate(meta_list):
        n_ctx = meta["n_ctx"]
        n_tgt = meta["n_tgt"]

        pred_norm = x0_pred[i, n_ctx:n_ctx + n_tgt, 0].cpu().numpy()
        true_norm = stec_full[i, n_ctx:n_ctx + n_tgt, 0].cpu().numpy()

        pred_orig = stec_norm.inverse_transform(pred_norm)
        true_orig = stec_norm.inverse_transform(true_norm)

        result_dfs.append(pd.DataFrame({
            "station_name":  meta["tgt_stations"],
            "ipp_latitude":  meta["tgt_lats"],
            "ipp_longitude": meta["tgt_lons"],
            "azimuth_deg":   meta["tgt_az"],
            "elevation_deg": meta["tgt_el"],
            "system_id":     meta["tgt_sys_ids"],
            "satellite_id":  meta["tgt_sat_ids"],
            "true_stec":     true_orig,
            "pred_stec":     pred_orig,
            "abs_error":     np.abs(pred_orig - true_orig),
        }))

    return result_dfs


def main():
    parser = argparse.ArgumentParser(description="STEC 条件扩散模型测试脚本（多星联合版）")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "configs", "default.yaml"),
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="指定 checkpoint 路径（不指定则自动查找 best checkpoint）"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="是否打印反向扩散进度"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 加载配置
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    print(f"[Config] 已加载：{args.config}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 系统过滤配置
    system_filter = cfg["data"].get("system_filter", None)
    system_ascii_code = get_system_ascii_code(system_filter) if system_filter else None
    if system_filter:
        print(f"[System] 卫星系统过滤：{system_filter}")
    else:
        print(f"[System] 全系统联合测试")

    # 设备
    device_str = cfg["training"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[Device] {device}")

    # ------------------------------------------------------------------
    # 1. 扫描历元文件 & 历元对齐
    # ------------------------------------------------------------------
    print("\n[1/5] 扫描并对齐历元文件...")
    model_stations_dir = os.path.join(project_root, cfg["data"]["model_stations_dir"])
    val_stations_dir   = os.path.join(project_root, cfg["data"]["val_stations_dir"])

    if not os.path.exists(model_stations_dir):
        print(f"[Error] model_stations 目录不存在：{model_stations_dir}")
        return
    if not os.path.exists(val_stations_dir):
        print(f"[Error] val_stations 目录不存在：{val_stations_dir}")
        return

    model_files = {f.stem: f for f in sorted(Path(model_stations_dir).glob("*.csv"))}
    val_files   = {f.stem: f for f in sorted(Path(val_stations_dir).glob("*.csv"))}

    # 找共同历元（文件名匹配）
    common_stems = sorted(set(model_files.keys()) & set(val_files.keys()))
    print(f"  model_stations: {len(model_files)} 个历元文件")
    print(f"  val_stations:   {len(val_files)} 个历元文件")
    print(f"  共同历元:       {len(common_stems)} 个")

    if not common_stems:
        print("[Error] 没有共同的历元文件，无法测试")
        return

    # 时间段过滤（文件名格式：20240218_000000-STEC，截取前 11 位即 YYYYMMDD_HH）
    test_time_start = cfg["inference"].get("test_time_start", None)
    test_time_end   = cfg["inference"].get("test_time_end", None)
    if test_time_start or test_time_end:
        filtered_stems = []
        for stem in common_stems:
            stem_hour = stem[:11]  # "20240218_00"
            if test_time_start and stem_hour < test_time_start:
                continue
            if test_time_end and stem_hour > test_time_end:
                continue
            filtered_stems.append(stem)
        print(f"  时间段过滤：{test_time_start or '不限'} ~ {test_time_end or '不限'}")
        print(f"  过滤后共同历元：{len(filtered_stems)} 个（原 {len(common_stems)} 个）")
        common_stems = filtered_stems

        if not common_stems:
            print("[Error] 时间段过滤后没有历元文件")
            return

    # 过滤：context IPP 数 < min_test_context_ipps 或 target IPP 数 < min_test_target_ipps
    min_ctx_ipps = cfg["data"].get("min_test_context_ipps", 20)
    min_tgt_ipps = cfg["data"].get("min_test_target_ipps", 10)

    valid_epochs = []
    for stem in common_stems:
        model_df_tmp = pd.read_csv(model_files[stem])
        val_df_tmp   = pd.read_csv(val_files[stem])
        if system_ascii_code is not None:
            model_df_tmp = model_df_tmp[model_df_tmp["system_id"] == system_ascii_code]
            val_df_tmp   = val_df_tmp[val_df_tmp["system_id"] == system_ascii_code]
        if len(model_df_tmp) >= min_ctx_ipps and len(val_df_tmp) >= min_tgt_ipps:
            valid_epochs.append(stem)

    print(f"  过滤后可用历元：{len(valid_epochs)} 个（context >= {min_ctx_ipps}, target >= {min_tgt_ipps}）")

    if not valid_epochs:
        print("[Error] 没有满足条件的历元，无法测试")
        return

    # ------------------------------------------------------------------
    # 2. 加载模型和归一化参数
    # ------------------------------------------------------------------
    print("\n[2/5] 加载模型和归一化参数...")

    # 构建与训练时相同的 setting_tag，定位实验目录
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

    norm_path = os.path.join(exp_dir, "normalizer.json")
    if not os.path.exists(norm_path):
        print(f"[Error] 归一化参数文件不存在：{norm_path}")
        return

    coord_norm, stec_norm, angle_norm = load_normalizers(norm_path)
    print(f"  已加载归一化参数：{norm_path}")

    # 确定 checkpoint 路径
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(exp_dir, "checkpoints", "ckpt_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[Error] checkpoint 不存在：{ckpt_path}")
        return

    model_cfg = dict(cfg["model"])
    model = build_model(model_cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"  已加载 checkpoint：{ckpt_path}")
    print(f"  训练轮次：{ckpt.get('epoch', '?')}  最佳验证 MAE：{ckpt.get('best_val_mae', float('nan')):.4f}")

    # ------------------------------------------------------------------
    # 3. 构建 SDE
    # ------------------------------------------------------------------
    print("\n[3/5] 构建 SDE...")
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

    # ------------------------------------------------------------------
    # 4. 推理预测（批量并行）
    # ------------------------------------------------------------------
    # 自动估算 batch size
    infer_batch_size = estimate_infer_batch_size(
        valid_epochs=valid_epochs,
        model_files=model_files,
        val_files=val_files,
        system_ascii_code=system_ascii_code,
        device=device,
        model_n_params=sum(p.numel() for p in model.parameters()),
        model_dim=cfg["model"].get("dim", 256),
        model_depth=cfg["model"].get("depth", 3),
    )

    n_epochs = len(valid_epochs)
    n_batches = (n_epochs + infer_batch_size - 1) // infer_batch_size
    print(f"\n[4/5] 推理预测（共 {n_epochs} 个历元，batch_size={infer_batch_size}，共 {n_batches} 个 batch）...")
    all_results = []
    n_done = 0

    for batch_idx in range(n_batches):
        batch_stems = valid_epochs[batch_idx * infer_batch_size : (batch_idx + 1) * infer_batch_size]

        # 读取并提取每个历元的数组
        epoch_data_list = []
        stem_list = []
        for stem in batch_stems:
            model_df_i = pd.read_csv(model_files[stem])
            val_df_i   = pd.read_csv(val_files[stem])
            try:
                data = _extract_epoch_arrays(model_df_i, val_df_i, system_ascii_code)
                epoch_data_list.append(data)
                stem_list.append(stem)
            except Exception as e:
                print(f"    [Warning] 历元 {stem} 数据提取失败：{e}")

        if not epoch_data_list:
            continue

        try:
            batch_tensors, meta_list = collate_inference_batch(
                epoch_data_list, coord_norm, stec_norm, angle_norm, device
            )
            result_dfs = predict_batch(
                model=model,
                sde=sde,
                batch_tensors=batch_tensors,
                meta_list=meta_list,
                stec_norm=stec_norm,
                device=device,
                verbose=args.verbose,
            )
            for stem, rdf in zip(stem_list, result_dfs):
                rdf["epoch_time"] = stem
                all_results.append(rdf)
        except Exception as e:
            print(f"    [Warning] batch {batch_idx+1}/{n_batches} 推理失败：{e}，逐历元回退...")
            for stem, data in zip(stem_list, epoch_data_list):
                try:
                    bt, ml = collate_inference_batch([data], coord_norm, stec_norm, angle_norm, device)
                    rdfs = predict_batch(model, sde, bt, ml, stec_norm, device, args.verbose)
                    rdfs[0]["epoch_time"] = stem
                    all_results.append(rdfs[0])
                except Exception as e2:
                    print(f"    [Warning] 历元 {stem} 推理失败：{e2}")

        n_done += len(stem_list)
        print(f"  [{batch_idx+1}/{n_batches}] 已完成 {n_done}/{n_epochs} 个历元")

    if not all_results:
        print("[Error] 所有历元推理均失败")
        return

    final_results = pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # 5. 导出结果
    # ------------------------------------------------------------------
    print(f"\n[5/5] 导出结果...")
    result_base = cfg["inference"].get("result_output_dir", "results/final_test")
    result_base_dir = os.path.dirname(result_base)  # "results"
    result_dir = os.path.join(project_root, result_base_dir, f"Test_{setting_tag}")
    os.makedirs(result_dir, exist_ok=True)

    # 导出完整结果（所有卫星，所有历元）
    if cfg["inference"].get("export_all_predictions", True):
        all_pred_path = os.path.join(result_dir, "all_predictions.csv")
        final_results.to_csv(all_pred_path, index=False)
        print(f"  完整预测结果：{all_pred_path}（{len(final_results)} 条记录）")

    # 整体指标（全局模式：汇总所有 target 点）
    mae_all  = float(final_results["abs_error"].mean())
    rmse_all = float(np.sqrt((final_results["abs_error"] ** 2).mean()))
    print(f"\n  整体指标（全局模式，汇总全部 target 点）：")
    print(f"    总 target 点数：{len(final_results)}")
    print(f"    MAE  = {mae_all:.4f} TECU")
    print(f"    RMSE = {rmse_all:.4f} TECU")

    # 按历元（样本）计算 per-sample RMSE/MAE，并导出 CSV
    print(f"\n  按历元分析指标：")
    epoch_metric_rows = []
    for epoch_stem, grp in final_results.groupby("epoch_time"):
        ep_mae  = float(grp["abs_error"].mean())
        ep_rmse = float(np.sqrt((grp["abs_error"] ** 2).mean()))
        n_pts   = len(grp)
        epoch_metric_rows.append({
            "epoch_time": epoch_stem,
            "n_target_points": n_pts,
            "mae_tecu": ep_mae,
            "rmse_tecu": ep_rmse,
        })

    epoch_metric_df = pd.DataFrame(epoch_metric_rows)
    epoch_metric_path = os.path.join(result_dir, "rmse_per_epoch.csv")
    epoch_metric_df.to_csv(epoch_metric_path, index=False)

    avg_sample_mae  = float(epoch_metric_df["mae_tecu"].mean())
    avg_sample_rmse = float(epoch_metric_df["rmse_tecu"].mean())
    print(f"    Per-sample 平均 MAE  = {avg_sample_mae:.4f} TECU")
    print(f"    Per-sample 平均 RMSE = {avg_sample_rmse:.4f} TECU")
    print(f"    Per-sample 指标 CSV：{epoch_metric_path}（{len(epoch_metric_df)} 个历元）")

    # 按 satellite_id + system_id 分组统计
    print(f"\n  按卫星分组指标：")
    summary_rows = []
    for (sys_id, sat_id), grp in final_results.groupby(["system_id", "satellite_id"]):
        mae_i  = float(grp["abs_error"].mean())
        rmse_i = float(np.sqrt((grp["abs_error"] ** 2).mean()))
        n_pts  = len(grp)
        print(f"    system_id={sys_id} satellite_id={sat_id:3d}: "
              f"MAE={mae_i:.4f}  RMSE={rmse_i:.4f}  N={n_pts}")
        summary_rows.append({
            "system_id":    sys_id,
            "satellite_id": sat_id,
            "n_points":     n_pts,
            "mae_tecu":     mae_i,
            "rmse_tecu":    rmse_i,
        })

    # 导出分组统计摘要
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(result_dir, "summary_by_satellite.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  分组统计摘要：{summary_path}")

    # ------------------------------------------------------------------
    # 按测站统计 IPP 建模精度
    # ------------------------------------------------------------------
    print(f"\n  按测站统计指标：")
    station_rows = []
    for station, grp in final_results.groupby("station_name"):
        mae_s  = float(grp["abs_error"].mean())
        rmse_s = float(np.sqrt((grp["abs_error"] ** 2).mean()))
        n_pts  = len(grp)
        print(f"    {station}: MAE={mae_s:.4f}  RMSE={rmse_s:.4f}  N={n_pts}")
        station_rows.append({
            "station_name": station,
            "n_points":     n_pts,
            "mae_tecu":     mae_s,
            "rmse_tecu":    rmse_s,
        })

    station_df = pd.DataFrame(station_rows)
    station_path = os.path.join(result_dir, "summary_by_station.csv")
    station_df.to_csv(station_path, index=False)
    print(f"  测站统计摘要：{station_path}")

    # ------------------------------------------------------------------
    # 按 30 分钟时段统计 IPP 建模精度
    # ------------------------------------------------------------------
    print(f"\n  按 30 分钟时段统计指标：")

    def stem_to_datetime(stem: str) -> datetime:
        ts = stem.split("-")[0]  # "20240218_000000"
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")

    final_results["datetime"] = final_results["epoch_time"].apply(stem_to_datetime)
    final_results["time_slot"] = final_results["datetime"].apply(
        lambda dt: dt.replace(minute=(dt.minute // 30) * 30, second=0).strftime("%Y%m%d_%H%M")
    )

    slot_rows = []
    for slot, grp in final_results.groupby("time_slot"):
        mae_t  = float(grp["abs_error"].mean())
        rmse_t = float(np.sqrt((grp["abs_error"] ** 2).mean()))
        n_pts  = len(grp)
        print(f"    {slot}: MAE={mae_t:.4f}  RMSE={rmse_t:.4f}  N={n_pts}")
        slot_rows.append({
            "time_slot":  slot,
            "n_points":   n_pts,
            "mae_tecu":   mae_t,
            "rmse_tecu":  rmse_t,
        })

    slot_df = pd.DataFrame(slot_rows)
    slot_path = os.path.join(result_dir, "summary_by_timeslot_30min.csv")
    slot_df.to_csv(slot_path, index=False)
    print(f"  时段统计摘要：{slot_path}")

    # 清理临时列
    final_results.drop(columns=["datetime", "time_slot"], inplace=True)

    # 导出整体指标 JSON
    metrics = {
        "n_epochs":            len(valid_epochs),
        "n_target_points":     len(final_results),
        "global_mae_tecu":     mae_all,
        "global_rmse_tecu":    rmse_all,
        "per_sample_mae_tecu": avg_sample_mae,
        "per_sample_rmse_tecu": avg_sample_rmse,
    }
    metrics_path = os.path.join(result_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  整体指标 JSON：{metrics_path}")

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()
