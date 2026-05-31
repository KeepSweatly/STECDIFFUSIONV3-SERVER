"""
plot/plot_error_scatter.py
===========================
绘制测试结果误差散点图。

横坐标：时间（历元）
纵坐标：每个 IPP 点的建模误差（|pred - true|, TECU）

使用方式：
    cd D:/Phd/IonoModeling/stec_diffusionv3
    python plot/plot_error_scatter.py
    python plot/plot_error_scatter.py --result_dir results/final_test
    python plot/plot_error_scatter.py --result_dir results/final_test_BDS
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stem_to_datetime(stem: str) -> datetime:
    ts = stem.split("-")[0]  # "20240218_000000"
    return datetime.strptime(ts, "%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(description="STEC 测试误差散点图")
    parser.add_argument(
        "--result_dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "final_test"
        ),
        help="测试结果目录（包含 all_predictions.csv）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出图片路径（默认保存到 result_dir/error_scatter.png）"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="输出图片 DPI"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3,
        help="散点透明度"
    )
    parser.add_argument(
        "--markersize", type=float, default=2.0,
        help="散点大小"
    )
    args = parser.parse_args()

    csv_path = os.path.join(args.result_dir, "all_predictions.csv")
    if not os.path.exists(csv_path):
        print(f"[Error] 预测结果文件不存在：{csv_path}")
        return

    print(f"[1/3] 读取预测结果：{csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  共 {len(df)} 条记录")

    df["datetime"] = df["epoch_time"].apply(stem_to_datetime)
    df["error"] = df["pred_stec"] - df["true_stec"]

    print("[2/3] 绘制误差散点图...")
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(
        df["datetime"], df["error"],
        s=args.markersize, alpha=args.alpha, c="steelblue", edgecolors="none",
    )

    ax.set_xlabel("Time (UTC)", fontsize=12)
    ax.set_ylabel("Error (pred - true, TECU)", fontsize=12)
    ax.set_title("STEC Prediction Error vs Time", fontsize=14)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    mae_val = df["abs_error"].mean()
    rmse_val = np.sqrt((df["error"] ** 2).mean())
    ax.axhline(mae_val, color="tomato", linestyle="--", linewidth=1.0, label=f"MAE = {mae_val:.3f} TECU")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = args.output or os.path.join(args.result_dir, "error_scatter.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[3/3] 图片已保存：{output_path}")
    print(f"  MAE  = {mae_val:.4f} TECU")
    print(f"  RMSE = {rmse_val:.4f} TECU")

    plt.close(fig)


if __name__ == "__main__":
    main()
