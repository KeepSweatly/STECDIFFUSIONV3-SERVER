"""
data/dataset.py
================
STEC 历元数据集类（多星联合版本）

概述：
    本模块实现按历元文件组织的 STEC 数据集，支持多卫星联合建模。
    每个历元文件（CSV）作为一个样本，包含该时刻所有测站和卫星的 IPP 点。

核心功能：
    1. 历元级样本组织
       - 每个 CSV 文件 = 一个样本（包含多颗卫星的 IPP 点）
       - 支持变长样本（不同历元的 IPP 点数量不同）
       - 自动过滤 IPP 点数过少的样本

    2. IPP 点级划分策略
       - 确定性打乱：使用 seed + idx 确保可复现
       - 80/20 划分：前 80% 作训练点池，后 20% 作验证点
       - 模式分离：训练/验证/测试使用不同的划分逻辑

    3. 三种数据模式
       train_context_target:
         - 只保留前 80% IPP 点作为训练点池
         - 训练时从中随机抽取 target（动态 mask）
         - 不返回 context_indices / target_indices

       val_eval:
         - 保留所有 IPP 点
         - 前 80% 作 context（预计算 indices）
         - 后 20% 作 target（预计算 indices）
         - 固定划分，确保评估一致性

       test_target:
         - 保留所有 IPP 点
         - 全部作为 target（待预测）
         - 返回 target_indices

    4. 数据归一化
       - 坐标归一化：(lat, lon) → 标准化空间
       - 角度归一化：(azimuth, elevation) → 标准化空间
       - STEC 归一化：原始 TECU → 标准化值
       - 训练集统计参数，验证集/测试集复用

    5. 样本过滤
       - 总 IPP 点数小于 min_points 的历元不参与训练/验证
       - batch 内按实际点数 padding 到同一长度，无截断

样本数据格式：
    {
        "epoch_time":          str                历元标识（文件名 stem）
        "source_file":         str                原始文件路径
        "coords":              np.ndarray [N, 2]  归一化后的 (lat, lon)
        "angles":              np.ndarray [N, 2]  归一化后的 (azimuth, elevation)
        "stec":                np.ndarray [N, 1]  归一化后的 STEC
        "system_ids":          np.ndarray [N]     卫星系统 ID（0=GPS, 1=GLONASS, ...）
        "satellite_ids":       np.ndarray [N]     卫星编号（用于结果分组）
        "station_names":       list[str]          测站名称列表
        "n_points":            int                有效点数 N
        "context_indices":     np.ndarray [Nc]    context 点索引（仅 val_eval 模式）
        "target_indices":      np.ndarray [Nt]    target 点索引（val_eval/test_target 模式）
    }

设计原则：
    - 多星联合：不同卫星系统的 IPP 点在同一样本中
    - IPP 点级划分：不依赖 station_name，支持任意点集
    - 模式分离：训练/验证/测试逻辑清晰分离
    - 可复现性：确定性打乱 + 固定 seed
    - 可扩展性：支持任意数量的卫星系统和测站

使用示例：
    # 构建训练集和验证集
    train_ds, val_ds, coord_norm, stec_norm, angle_norm = build_train_val_datasets(
        model_stations_dir="data/model_stations",
        cfg=config["data"],
    )

    # 获取单个样本
    sample = train_ds[0]
    print(f"历元: {sample['epoch_time']}")
    print(f"IPP 点数: {sample['n_points']}")
    print(f"坐标形状: {sample['coords'].shape}")
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional

from utils.normalizer import CoordNormalizer, STECNormalizer


def _compute_cache_key(epoch_files: List[Path], cfg: dict) -> str:
    """根据文件列表的修改时间和配置参数生成缓存键"""
    h = hashlib.md5()
    for f in sorted(epoch_files):
        stat = os.stat(f)
        h.update(f"{f}:{stat.st_mtime}:{stat.st_size}".encode())
    # 纳入影响数据处理的配置项
    key_cfg = {
        "split_ratio": cfg.get("split_ratio", 0.8),
        "min_points": cfg.get("min_points", 50),
        "seed": cfg.get("seed", 2026),
        "system_filter": cfg.get("system_filter", None),
    }
    h.update(str(sorted(key_cfg.items())).encode())
    return h.hexdigest()[:16]


# ======================================================================
# System ID 映射工具
# ======================================================================

def map_system_id_to_index(system_ids: np.ndarray) -> np.ndarray:
    """
    将 GNSS system_id (ASCII 码) 映射到模型可用的索引 (0-9)。

    常见的 GNSS 系统 ASCII 码：
        'G' (71) = GPS
        'R' (82) = GLONASS
        'E' (69) = Galileo
        'C' (67) = BDS (北斗)
        'J' (74) = QZSS
        'I' (73) = IRNSS

    映射规则：
        0: padding (保留)
        1: GPS (G=71)
        2: GLONASS (R=82)
        3: Galileo (E=69)
        4: BDS (C=67)
        5: QZSS (J=74)
        6: IRNSS (I=73)
        7-9: 保留给未来系统

    Args:
        system_ids: 原始 system_id 数组 (ASCII 码)

    Returns:
        mapped_ids: 映射后的索引数组 (0-9)
    """
    # 创建映射字典
    mapping = {
        71: 1,  # G -> GPS
        82: 2,  # R -> GLONASS
        69: 3,  # E -> Galileo
        67: 4,  # C -> BDS
        74: 5,  # J -> QZSS
        73: 6,  # I -> IRNSS
    }

    # 向量化映射
    mapped_ids = np.zeros_like(system_ids)
    for ascii_code, idx in mapping.items():
        mapped_ids[system_ids == ascii_code] = idx

    # 检查是否有未知的 system_id
    unknown_mask = (mapped_ids == 0) & (system_ids != 0)
    if unknown_mask.any():
        unknown_ids = np.unique(system_ids[unknown_mask])
        print(f"[Warning] Unknown system_id values: {unknown_ids} (ASCII: {[chr(x) for x in unknown_ids]})")
        # 将未知系统映射到索引 7
        mapped_ids[unknown_mask] = 7

    return mapped_ids


# 系统名称到 ASCII 码的映射
SYSTEM_NAME_TO_ASCII = {
    "GPS": 71,
    "GLONASS": 82,
    "Galileo": 69,
    "BDS": 67,
    "QZSS": 74,
    "IRNSS": 73,
}


def get_system_ascii_code(system_name: str) -> int:
    """
    将用户友好的系统名称映射为 CSV 中的 ASCII 码。

    Args:
        system_name: "GPS", "BDS", "Galileo", "GLONASS", "QZSS", "IRNSS"

    Returns:
        对应的 ASCII 码（如 GPS → 71, BDS → 67）
    """
    if system_name not in SYSTEM_NAME_TO_ASCII:
        raise ValueError(
            f"未知的卫星系统名称: '{system_name}'，"
            f"支持的系统: {list(SYSTEM_NAME_TO_ASCII.keys())}"
        )
    return SYSTEM_NAME_TO_ASCII[system_name]


class STECEpochDataset(Dataset):
    """
    按历元文件组织的 STEC 数据集（IPP 点级划分版本）。

    Args:
        epoch_files (List[Path]): 历元文件路径列表
        mode (str): 数据使用模式
            - "train_context_target": 训练模式，只保留前 80% IPP 点作为训练点池
            - "val_eval": 验证模式，前 80% 作 context，后 20% 作 target
            - "test_target": 测试模式，全部 IPP 点作为 target
        split_ratio (float): IPP 点划分比例（默认 0.8）
        min_points (int): 历元总 IPP 点数下限，低于此值的历元不参与训练/验证
        coord_normalizer (CoordNormalizer): 若为 None 则从当前数据统计
        stec_normalizer (STECNormalizer): 若为 None 则从当前数据统计
        angle_normalizer (CoordNormalizer): 角度归一化器（复用 CoordNormalizer）
        seed (int): 随机种子
    """

    def __init__(
        self,
        epoch_files: List[Path],
        mode: str = "train_context_target",
        split_ratio: float = 0.8,
        min_points: int = 50,
        coord_normalizer: Optional[CoordNormalizer] = None,
        stec_normalizer: Optional[STECNormalizer] = None,
        angle_normalizer: Optional[CoordNormalizer] = None,
        seed: int = 2026,
        system_filter: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        assert mode in ("train_context_target", "val_eval", "test_target"), \
            f"mode 必须为 'train_context_target', 'val_eval' 或 'test_target'，得到 {mode}"

        self.mode = mode
        self.split_ratio = split_ratio
        self.seed = seed
        self.system_filter = system_filter
        self.system_ascii_code = get_system_ascii_code(system_filter) if system_filter else None

        sys_info = f"（系统过滤: {system_filter}）" if system_filter else "（全系统）"

        # ---------- 尝试从缓存加载 ----------
        cache_key = _compute_cache_key(epoch_files, {
            "split_ratio": split_ratio, "min_points": min_points,
            "seed": seed, "system_filter": system_filter,
        })
        cache_file = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"dataset_{mode}_{cache_key}.pkl")

        if cache_file is not None and os.path.exists(cache_file):
            print(f"[Dataset-{mode}] {sys_info} 发现缓存，直接加载：{cache_file}")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            self.valid_files = cached["valid_files"]
            self.coord_normalizer = cached["coord_normalizer"]
            self.stec_normalizer = cached["stec_normalizer"]
            self.angle_normalizer = cached["angle_normalizer"]
            # 若调用方传入了 normalizer，以传入的为准（val 复用 train 的）
            if coord_normalizer is not None:
                self.coord_normalizer = coord_normalizer
            if stec_normalizer is not None:
                self.stec_normalizer = stec_normalizer
            if angle_normalizer is not None:
                self.angle_normalizer = angle_normalizer
            print(f"[Dataset-{mode}] {sys_info} 缓存加载完成，共 {len(self.valid_files)} 个历元")
            return

        # ---------- 1. 过滤样本：总 IPP 数 < min_points ----------
        print(f"[Dataset-{mode}] {sys_info} 未找到缓存，从原始 CSV 构建数据集...")
        self.valid_files = []
        for fpath in epoch_files:
            df = pd.read_csv(fpath)
            if self.system_ascii_code is not None:
                df = df[df["system_id"] == self.system_ascii_code]
            if len(df) >= min_points:
                self.valid_files.append(fpath)

        print(f"[Dataset-{mode}] {sys_info} 过滤前：{len(epoch_files)} 个历元文件")
        print(f"[Dataset-{mode}] {sys_info} 过滤后：{len(self.valid_files)} 个历元文件（IPP 数 >= {min_points}）")

        if len(self.valid_files) == 0:
            raise ValueError(f"模式 {mode} 的数据集为空，请检查过滤阈值或数据目录")

        # ---------- 2. 统计归一化参数（仅在 normalizer 为 None 时） ----------
        if coord_normalizer is None or stec_normalizer is None or angle_normalizer is None:
            print(f"[Dataset-{mode}] 统计归一化参数...")
            all_coords = []
            all_stec = []
            all_angles = []

            for fpath in self.valid_files:
                df = pd.read_csv(fpath)
                if self.system_ascii_code is not None:
                    df = df[df["system_id"] == self.system_ascii_code]
                all_coords.append(df[["ipp_latitude", "ipp_longitude"]].values)
                all_stec.append(df["stec"].values)
                all_angles.append(df[["azimuth_deg", "elevation_deg"]].values)

            all_coords = np.vstack(all_coords)
            all_stec = np.concatenate(all_stec)
            all_angles = np.vstack(all_angles)

            if coord_normalizer is None:
                coord_normalizer = CoordNormalizer()
                coord_normalizer.fit(all_coords[:, 0], all_coords[:, 1])

            if stec_normalizer is None:
                stec_normalizer = STECNormalizer()
                stec_normalizer.fit(all_stec)

            if angle_normalizer is None:
                angle_normalizer = CoordNormalizer()
                angle_normalizer.fit(all_angles[:, 0], all_angles[:, 1])

        self.coord_normalizer = coord_normalizer
        self.stec_normalizer = stec_normalizer
        self.angle_normalizer = angle_normalizer

        # ---------- 3. 保存缓存（仅保存 train 模式的 normalizer，val 复用） ----------
        if cache_file is not None:
            cached = {
                "valid_files": self.valid_files,
                "coord_normalizer": self.coord_normalizer,
                "stec_normalizer": self.stec_normalizer,
                "angle_normalizer": self.angle_normalizer,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cached, f)
            print(f"[Dataset-{mode}] 数据集缓存已保存：{cache_file}")

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> dict:
        """
        返回单个历元样本（变长，尚未 padding）。
        """
        fpath = self.valid_files[idx]
        df = pd.read_csv(fpath)

        # 按系统过滤
        if self.system_ascii_code is not None:
            df = df[df["system_id"] == self.system_ascii_code].reset_index(drop=True)

        # 确定性打乱（避免文件原始顺序偏置）
        rng = np.random.default_rng(self.seed + idx)
        perm = rng.permutation(len(df))  # 这里的df是单个历元的所有IPP点，打乱这些点的顺序，先打乱再划分，包括后续的训练点池以及验证点池，甚至是后续的target和context
        df = df.iloc[perm].reset_index(drop=True)

        # 根据 mode 切分数据
        n_total = len(df)
        n_train = int(n_total * self.split_ratio)

        if self.mode == "train_context_target":
            # 只保留前 80% 作为训练点池
            df = df.iloc[:n_train].reset_index(drop=True)
            context_indices = None
            target_indices = None

        elif self.mode == "val_eval":
            # 前 80% 作 context，后 20% 作 target
            context_indices = np.arange(n_train)
            target_indices = np.arange(n_train, n_total)

        elif self.mode == "test_target":
            # 全部作为 target
            context_indices = None
            target_indices = np.arange(n_total)

        # 提取字段
        coords = df[["ipp_latitude", "ipp_longitude"]].values.astype(np.float32)
        angles = df[["azimuth_deg", "elevation_deg"]].values.astype(np.float32)
        stec = df["stec"].values.astype(np.float32)
        system_ids = df["system_id"].values.astype(np.int64)
        # 映射 system_id 从 ASCII 码到模型索引 (0-9)
        system_ids = map_system_id_to_index(system_ids)
        satellite_ids = df["satellite_id"].values.astype(np.int64)
        station_names = df["station_name"].tolist()

        n = len(df)

        # 归一化
        coords_norm = self.coord_normalizer.transform(coords[:, 0], coords[:, 1])  # [N, 2]
        angles_norm = self.angle_normalizer.transform(angles[:, 0], angles[:, 1])  # [N, 2]
        stec_norm = self.stec_normalizer.transform(stec)  # [N]
        stec_norm = stec_norm[:, np.newaxis]  # [N, 1]

        result = {
            "epoch_time": fpath.stem,
            "source_file": str(fpath),
            "coords": coords_norm,
            "angles": angles_norm,
            "stec": stec_norm,
            "system_ids": system_ids,
            "satellite_ids": satellite_ids,
            "station_names": station_names,
            "n_points": n,
        }

        # 仅 val_eval 和 test_target 模式返回 indices
        if context_indices is not None:
            result["context_indices"] = context_indices
        if target_indices is not None:
            result["target_indices"] = target_indices

        return result


def build_train_val_datasets(
    model_stations_dir: str,
    cfg: dict,
    cache_dir: Optional[str] = None,
) -> tuple:
    """
    构建训练集和验证集，共享归一化参数。

    Args:
        model_stations_dir: model_stations 目录路径
        cfg: 配置字典（来自 default.yaml 的 data 节）
        cache_dir: 缓存目录路径，None 则不使用缓存

    Returns:
        train_dataset, val_dataset, coord_normalizer, stec_normalizer, angle_normalizer
    """
    epoch_files = sorted(Path(model_stations_dir).glob("*.csv"))

    split_ratio = cfg.get("split_ratio", 0.8)
    min_points = cfg.get("min_points", 50)
    seed = cfg.get("seed", 2026)
    system_filter = cfg.get("system_filter", None)

    # 先构建训练集，从中统计归一化参数
    train_ds = STECEpochDataset(
        epoch_files=epoch_files,
        mode="train_context_target",
        split_ratio=split_ratio,
        min_points=min_points,
        coord_normalizer=None,
        stec_normalizer=None,
        angle_normalizer=None,
        seed=seed,
        system_filter=system_filter,
        cache_dir=cache_dir,
    )

    # 验证集复用训练集的归一化参数
    val_ds = STECEpochDataset(
        epoch_files=epoch_files,
        mode="val_eval",
        split_ratio=split_ratio,
        min_points=min_points,
        coord_normalizer=train_ds.coord_normalizer,
        stec_normalizer=train_ds.stec_normalizer,
        angle_normalizer=train_ds.angle_normalizer,
        seed=seed,
        system_filter=system_filter,
        cache_dir=cache_dir,
    )

    return (
        train_ds,
        val_ds,
        train_ds.coord_normalizer,
        train_ds.stec_normalizer,
        train_ds.angle_normalizer,
    )
