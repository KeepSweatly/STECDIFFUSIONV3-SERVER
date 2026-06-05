#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 IPP 点的 STEC 建模情况, 共 4 个子图 (2x2):

  (a) 模型预测 STEC (pred_stec) 散点图
  (b) 真实 STEC    (true_stec) 散点图
  (c) 建模误差     (abs_error) 散点图
  (d) 建模误差直方图 (误差统计分布)

约定:
  - 颜色 (colorbar)  -> 预测/真实 STEC 大小 或 误差大小
  - 散点标志 (marker) -> 不同测站 (station)
  - 仅绘制指定测试历元的 IPP 点

注意: 本脚本位于 plots/ 目录下, 读取 results/ 数据时基于脚本所在位置
      推算项目根目录, 因此无论从哪个工作目录运行都能正确找到数据。
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------
# 路径处理: plots/plot_ipp_stec.py  ->  项目根目录是其上一级
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../plots
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                   # 项目根
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")               # plots/figures

DEFAULT_CSV = os.path.join(
    PROJECT_ROOT,
    "results",
    "Test_chp_joint_BDS_d1024_L4_h8_mr4.0_sig2.0_g2.0_lr0.0001_"
    "bs32_ik20_ip2.0_mmi0.1_mma0.3_wcd0.2",
    "all_predictions.csv",
)

# 不同测站标志 (marker)。测站数较多时循环使用。
MARKER_POOL = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]


def parse_args():
    p = argparse.ArgumentParser(
        description="绘制 IPP 点 STEC 建模情况 (4 子图)"
    )
    p.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="all_predictions.csv 的路径 (默认指向 d1024 结果)",
    )
    p.add_argument(
        "--epoch",
        default=None,
        help="要绘制的测试历元, 例如 20240218_120000 或 "
        "20240218_120000-STEC。默认取数据中的第一个历元。",
    )
    p.add_argument(
        "--out",
        default=None,
        help="输出图片路径。默认存到 plots/figures/ 下, "
        "文件名含历元。",
    )
    p.add_argument(
        "--err-vmax",
        type=float,
        default=None,
        help="误差子图 colorbar 上限 (TECU), 默认取该历元误差的 95 分位",
    )
    p.add_argument(
        "--bins", type=int, default=30, help="误差直方图分箱数"
    )
    p.add_argument(
        "--dpi", type=int, default=300, help="输出图片 DPI"
    )
    return p.parse_args()


def normalize_epoch(epoch_raw):
    """把用户输入的历元统一成 csv 中的写法 (带 -STEC 后缀)。"""
    if epoch_raw is None:
        return None
    epoch_raw = epoch_raw.strip()
    if epoch_raw.endswith("-STEC"):
        return epoch_raw
    return epoch_raw + "-STEC"


def load_epoch(csv_path, epoch):
    """只载入指定历元的数据, 避免把上百万行全部读进内存。"""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    if epoch is None:
        head = pd.read_csv(csv_path, nrows=1)
        epoch = head["epoch_time"].iloc[0]
        print(f"[info] 未指定历元, 使用默认首个历元: {epoch}")

    chunks = []
    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        total += len(chunk)
        sub = chunk[chunk["epoch_time"] == epoch]
        if not sub.empty:
            chunks.append(sub)
        hit = sum(len(c) for c in chunks)
        print(f"  [扫描] 已处理 {total} 行, 命中 {hit} 行", end="\r")
    print()

    if not chunks:
        raise ValueError(f"历元 {epoch} 在文件中没有数据, 请检查输入。")

    df = pd.concat(chunks, ignore_index=True)
    print(f"[info] 历元 {epoch} 共 {len(df)} 个 IPP 点, "
          f"测站 {df['station_name'].nunique()} 个")
    return df, epoch


def make_marker_map(stations):
    """为每个测站分配一个固定 marker。"""
    return {
        st: MARKER_POOL[i % len(MARKER_POOL)]
        for i, st in enumerate(stations)
    }


def scatter_subplot(ax, df, value_col, marker_map, cmap,
                    vmin, vmax, cbar_label, title):
    """在 ax 上绘制散点子图: 颜色=value_col 大小, marker=测站。"""
    sc = None
    for st, marker in marker_map.items():
        sub = df[df["station_name"] == st]
        if sub.empty:
            continue
        sc = ax.scatter(
            sub["ipp_longitude"],
            sub["ipp_latitude"],
            c=sub[value_col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=marker,
            s=55,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )

    ax.set_xlabel("IPP Longitude (deg)", fontsize=11)
    ax.set_ylabel("IPP Latitude (deg)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    return sc


def hist_subplot(ax, df, bins, title):
    """在 ax 上绘制建模误差直方图, 并标注统计量。"""
    err = df["abs_error"].values
    ax.hist(
        err, bins=bins, color="#4C72B0",
        edgecolor="black", linewidth=0.5, alpha=0.85,
    )

    mean_e = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    p95 = float(np.percentile(err, 95))

    ax.axvline(mean_e, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_e:.3f}")
    ax.axvline(rmse, color="green", linestyle="-.", linewidth=1.5,
               label=f"RMSE = {rmse:.3f}")
    ax.axvline(p95, color="purple", linestyle=":", linewidth=1.5,
               label=f"P95 = {p95:.3f}")

    ax.set_xlabel("Modeling Abs. Error (TECU)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, framealpha=0.9)


def build_station_legend(fig, marker_map):
    """用统一的灰色 marker 表示测站, 单独做图例 (与颜色解耦)。"""
    handles = [
        Line2D(
            [0], [0], marker=marker, color="none",
            markerfacecolor="gray", markeredgecolor="black",
            markersize=9, linestyle="None", label=st,
        )
        for st, marker in marker_map.items()
    ]
    fig.legend(
        handles=handles,
        title="Station",
        loc="center right",
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9,
        bbox_to_anchor=(1.0, 0.5),
    )


def main():
    args = parse_args()
    epoch = normalize_epoch(args.epoch)

    df, epoch_used = load_epoch(args.csv, epoch)
    epoch_label = epoch_used.replace("-STEC", "")

    stations = sorted(df["station_name"].unique())
    marker_map = make_marker_map(stations)

    # 误差 colorbar 上限: 默认取 95 分位, 防止极端值压扁色彩
    if args.err_vmax is not None:
        err_vmax = args.err_vmax
    else:
        err_vmax = float(np.percentile(df["abs_error"], 95))
    err_vmin = 0.0

    # 预测与真实 STEC 共用同一色标范围, 便于直接对比两子图
    stec_vmin = float(min(df["true_stec"].min(), df["pred_stec"].min()))
    stec_vmax = float(max(df["true_stec"].max(), df["pred_stec"].max()))

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # (a) 预测 STEC
    scatter_subplot(
        axes[0, 0], df, "pred_stec", marker_map,
        cmap="viridis", vmin=stec_vmin, vmax=stec_vmax,
        cbar_label="Predicted STEC (TECU)",
        title=f"(a) IPP Predicted STEC  @ {epoch_label}",
    )

    # (b) 真实 STEC
    scatter_subplot(
        axes[0, 1], df, "true_stec", marker_map,
        cmap="viridis", vmin=stec_vmin, vmax=stec_vmax,
        cbar_label="True STEC (TECU)",
        title=f"(b) IPP True STEC  @ {epoch_label}",
    )

    # (c) 建模误差散点
    scatter_subplot(
        axes[1, 0], df, "abs_error", marker_map,
        cmap="YlOrRd", vmin=err_vmin, vmax=err_vmax,
        cbar_label="Modeling Abs. Error (TECU)",
        title=f"(c) IPP STEC Modeling Error  @ {epoch_label}",
    )

    # (d) 误差直方图
    hist_subplot(
        axes[1, 1], df, args.bins,
        title=f"(d) Modeling Error Histogram  @ {epoch_label}",
    )

    # 测站图例 (整图右侧)
    build_station_legend(fig, marker_map)

    fig.suptitle(
        f"IPP STEC Modeling Overview  (epoch: {epoch_label}, "
        f"{len(df)} points, {len(stations)} stations)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])

    # 输出路径: 默认存到 plots/figures/
    if args.out is not None:
        out_path = args.out
    else:
        os.makedirs(FIG_DIR, exist_ok=True)
        out_path = os.path.join(
            FIG_DIR, f"ipp_stec_{epoch_label}.png"
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[done] 已保存图片: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
