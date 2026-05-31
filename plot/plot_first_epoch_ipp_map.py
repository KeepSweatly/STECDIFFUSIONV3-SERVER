"""
Plot IPP locations for the first common epoch in model_stations and val_stations.

The two datasets share the same STEC color mapping while using different markers:
    - model_stations: circles
    - val_stations: triangles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


def find_first_common_epoch(model_dir: Path, val_dir: Path) -> tuple[Path, Path]:
    model_files = {p.name: p for p in sorted(model_dir.glob("*.csv"))}
    val_files = {p.name: p for p in sorted(val_dir.glob("*.csv"))}
    common_names = sorted(set(model_files) & set(val_files))

    if not common_names:
        raise FileNotFoundError("No common epoch CSV files found between model_stations and val_stations.")

    first_name = common_names[0]
    return model_files[first_name], val_files[first_name]


def build_colormap() -> LinearSegmentedColormap:
    colors = [
        "#143d59",
        "#1f6f8b",
        "#2ab7ca",
        "#f4d35e",
        "#ee964b",
        "#d1495b",
    ]
    return LinearSegmentedColormap.from_list("stec_map", colors, N=256)


def plot_epoch(model_df: pd.DataFrame, val_df: pd.DataFrame, epoch_name: str, output_path: Path) -> None:
    combined_stec = pd.concat([model_df["stec"], val_df["stec"]], ignore_index=True)
    norm = Normalize(vmin=float(combined_stec.min()), vmax=float(combined_stec.max()))
    cmap = build_colormap()

    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
    fig.patch.set_facecolor("#f7f7f5")
    ax.set_facecolor("#fbfbfa")

    scatter_model = ax.scatter(
        model_df["ipp_longitude"],
        model_df["ipp_latitude"],
        c=model_df["stec"],
        cmap=cmap,
        norm=norm,
        s=44,
        marker="o",
        edgecolors="#16324f",
        linewidths=0.55,
        alpha=0.92,
        zorder=3,
    )

    ax.scatter(
        val_df["ipp_longitude"],
        val_df["ipp_latitude"],
        c=val_df["stec"],
        cmap=cmap,
        norm=norm,
        s=62,
        marker="^",
        edgecolors="#4a1c40",
        linewidths=0.7,
        alpha=0.96,
        zorder=4,
    )

    cbar = fig.colorbar(scatter_model, ax=ax, pad=0.018, fraction=0.045)
    cbar.set_label("STEC", fontsize=12, color="#283044")
    cbar.outline.set_edgecolor("#9aa6b2")
    cbar.ax.tick_params(labelsize=10, colors="#4b5563")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=f"model_stations ({len(model_df)} points)",
            markerfacecolor="#5bbfd2",
            markeredgecolor="#16324f",
            markeredgewidth=0.9,
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="",
            label=f"val_stations ({len(val_df)} points)",
            markerfacecolor="#f08a5d",
            markeredgecolor="#4a1c40",
            markeredgewidth=0.9,
            markersize=8,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        facecolor="#ffffff",
        edgecolor="#d4d4d8",
        fontsize=10,
    )

    ax.set_title("IPP Distribution For First Common Epoch", fontsize=18, color="#1f2937", pad=14)
    ax.text(
        0.02,
        0.98,
        f"Epoch: {epoch_name}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#d9d9d9", alpha=0.95),
    )

    ax.set_xlabel("IPP Longitude", fontsize=12, color="#283044")
    ax.set_ylabel("IPP Latitude", fontsize=12, color="#283044")
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.26, color="#64748b")

    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.0)

    ax.tick_params(axis="both", labelsize=10, colors="#334155")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot IPP points for the first common epoch in model_stations and val_stations."
    )
    parser.add_argument("--model-dir", type=Path, default=Path("model_stations"))
    parser.add_argument("--val-dir", type=Path, default=Path("val_stations"))
    parser.add_argument("--output", type=Path, default=Path("plot") / "first_epoch_ipp_map.png")
    args = parser.parse_args()

    model_file, val_file = find_first_common_epoch(args.model_dir, args.val_dir)
    model_df = pd.read_csv(model_file)
    val_df = pd.read_csv(val_file)

    required_cols = {"ipp_latitude", "ipp_longitude", "stec"}
    missing_model = required_cols - set(model_df.columns)
    missing_val = required_cols - set(val_df.columns)
    if missing_model:
        raise ValueError(f"Missing required columns in {model_file}: {sorted(missing_model)}")
    if missing_val:
        raise ValueError(f"Missing required columns in {val_file}: {sorted(missing_val)}")

    plot_epoch(model_df, val_df, model_file.stem, args.output)
    print(f"Saved figure to: {args.output}")
    print(f"Model epoch file: {model_file}")
    print(f"Val epoch file:   {val_file}")


if __name__ == "__main__":
    main()
