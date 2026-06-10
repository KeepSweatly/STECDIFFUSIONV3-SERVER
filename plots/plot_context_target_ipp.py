#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化指定历元下 context 与 target 的 IPP STEC 分布 (单张图)。

  - context (model_stations): 圆形 marker "o"
  - target  (val_stations):   三角 marker "^"
  - 两类点共用同一 STEC 色标 (colorbar), 便于直接对比量级
  - 不再按测站区分形状, 只用形状区分 context / target

数据来源:
  model_stations / val_stations 目录下, 每个历元一个 CSV,
  文件名形如 20240218_120000-STEC.csv, 列含
  ipp_latitude, ipp_longitude, stec, station_name 等。

注意: 本脚本位于 plots/ 下。项目里 model_stations / val_stations 是指向
      真实数据目录的快捷方式, 实际数据在 D:/Phd/IonoModeling/dataset/ 下的
      model_stations_demo_IPP / val_stations_demo_IPP。脚本默认解析到真实
      数据目录, 也可用 --model-dir / --val-dir 覆盖。
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------
# 路径处理: plots/plot_context_target_ipp.py -> 项目根是其上一级
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../plots
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                   # 项目根
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")               # plots/figures

# 真实数据目录 (快捷方式指向处)
# PROJECT_ROOT = D:/Phd/IonoModeling/stec_diffusionv3-server
# 其上一级 D:/Phd/IonoModeling 下的 dataset 即真实数据根目录
DATASET_ROOT = os.path.join(
    os.path.dirname(PROJECT_ROOT), "dataset"
)  # D:/Phd/IonoModeling/dataset
DEFAULT_MODEL_DIR = os.path.join(DATASET_ROOT, "model_stations_demo_IPP")
DEFAULT_VAL_DIR = os.path.join(DATASET_ROOT, "val_stations_demo_IPP")


def parse_args():
    p = argparse.ArgumentParser(
        description="可视化指定历元下 context/target 的 IPP STEC 分布"
    )
    p.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="context (model_stations) 历元 CSV 所在目录",
    )
    p.add_argument(
        "--val-dir",
        default=DEFAULT_VAL_DIR,
        help="target (val_stations) 历元 CSV 所在目录",
    )
    p.add_argument(
        "--epoch",
        default=None,
        help="要绘制的历元, 例如 20240218_120000 或 "
        "20240218_120000-STEC。默认取两目录共有的第一个历元。",
    )
    p.add_argument(
        "--out",
        default=None,
        help="输出图片路径。默认存到 plots/figures/ 下, 文件名含历元。",
    )
    p.add_argument("--dpi", type=int, default=300, help="输出图片 DPI")
    return p.parse_args()


def epoch_to_filename(epoch_raw):
    """把历元统一成 CSV 文件名 (带 -STEC.csv)。"""
    epoch_raw = epoch_raw.strip()
    if epoch_raw.endswith(".csv"):
        return epoch_raw
    if epoch_raw.endswith("-STEC"):
        return epoch_raw + ".csv"
    return epoch_raw + "-STEC.csv"


def resolve_epoch_files(model_dir, val_dir, epoch):
    """确定 context / target 两个历元文件的路径。"""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"找不到 context 目录: {model_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"找不到 target 目录: {val_dir}")

    if epoch is None:
        # 取两目录共有的第一个历元文件
        model_files = {os.path.basename(p)
                       for p in glob.glob(os.path.join(model_dir, "*.csv"))}
        val_files = {os.path.basename(p)
                     for p in glob.glob(os.path.join(val_dir, "*.csv"))}
        common = sorted(model_files & val_files)
        if not common:
            raise FileNotFoundError(
                "context / target 目录之间没有共同历元文件。"
            )
        fname = common[0]
        print(f"[info] 未指定历元, 使用首个共有历元: {fname}")
    else:
        fname = epoch_to_filename(epoch)

    model_path = os.path.join(model_dir, fname)
    val_path = os.path.join(val_dir, fname)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"context 缺少该历元: {model_path}")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"target 缺少该历元: {val_path}")
    return model_path, val_path, fname


def main():
    args = parse_args()

    model_path, val_path, fname = resolve_epoch_files(
        args.model_dir, args.val_dir, args.epoch
    )
    epoch_label = fname.replace("-STEC.csv", "").replace(".csv", "")

    model_df = pd.read_csv(model_path)
    val_df = pd.read_csv(val_path)

    required = {"ipp_latitude", "ipp_longitude", "stec"}
    for name, df in [("context", model_df), ("target", val_df)]:
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"{name} 数据缺少列: {sorted(miss)}")

    print(f"[info] 历元 {epoch_label}: context {len(model_df)} 点, "
          f"target {len(val_df)} 点")

    # 两类点共用同一 STEC 色标范围
    combined = pd.concat([model_df["stec"], val_df["stec"]],
                         ignore_index=True)
    norm = Normalize(vmin=float(combined.min()), vmax=float(combined.max()))
    cmap = "viridis"

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(12, 8))

    # context: 圆形
    sc = ax.scatter(
        model_df["ipp_longitude"], model_df["ipp_latitude"],
        c=model_df["stec"], cmap=cmap, norm=norm,
        marker="o", s=55, edgecolors="black", linewidths=0.4,
        alpha=0.9, zorder=3,
    )
    # target: 三角 (略大, 突出)
    ax.scatter(
        val_df["ipp_longitude"], val_df["ipp_latitude"],
        c=val_df["stec"], cmap=cmap, norm=norm,
        marker="^", s=90, edgecolors="black", linewidths=0.7,
        alpha=0.95, zorder=4,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("STEC (TECU)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # 形状图例 (与颜色解耦, 用灰色统一表示)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=9,
               label=f"context / model_stations ({len(model_df)} pts)"),
        Line2D([0], [0], marker="^", linestyle="None",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=11,
               label=f"target / val_stations ({len(val_df)} pts)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10,
              framealpha=0.92, title="Point type", title_fontsize=10)

    ax.set_xlabel("IPP Longitude (deg)", fontsize=12)
    ax.set_ylabel("IPP Latitude (deg)", fontsize=12)
    ax.set_title(
        f"IPP STEC: Context vs Target  @ {epoch_label}",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)

    fig.tight_layout()

    if args.out is not None:
        out_path = args.out
    else:
        os.makedirs(FIG_DIR, exist_ok=True)
        out_path = os.path.join(
            FIG_DIR, f"ipp_context_target_{epoch_label}.png"
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[done] 已保存图片: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
