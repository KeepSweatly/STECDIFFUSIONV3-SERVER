"""
inference/product_eval.py
==========================
格网产品（product / grid-product）推理评估流程。

科学含义：
    模拟真实业务场景——服务端不知道用户站点位置，只能播发一个规则格网产品。
    用户根据自身 IPP 的 (latitude, longitude, azimuth, elevation) 从格网产品
    插值得到自己的 STEC。本模块评估「格网产品播发 + 用户端插值」的实际精度。

与 direct evaluation 的区别：
    - direct ：模型直接以 val_stations 的 IPP 点为 target，预测其 STEC。
    - product：模型以规则 4D 格网点为 target，生成格网 STEC；val_stations 的
               预测值由其所在 4D cell 的 16 个角点 IDW 插值得到。
    两者 label 都用 val_stations 真实 STEC，可直接对比。

核心约束（严格遵守）：
    1. val_stations 的 IPP 点绝不作为模型 target 直接预测；
    2. val_stations 的 STEC label 只用于最终评估，不参与格网预测与 IDW；
    3. 模型只对格网点（或评估所需角点）预测 STEC；
    4. 每个历元单独建格网，格网不区分卫星；
    5. 4D 格网默认用所在 cell 的全部 16 个角点做 IDW；
    6. 不保存完整格网产品，不保存 all_predictions（内存友好）。

4D 格网：
    维度顺序固定为 (lat, lon, az, el)。每个 val 点定位到一个 4D cell，
    该 cell 有 2^4 = 16 个角点。IDW 距离在「归一化后」的 4D 空间计算，
    避免 lat/lon/az/el 量纲差异导致某一维主导。

内存友好策略：
    - corner-only 优化：只预测 val 点所在 cell 的去重角点，而非整个格网；
    - 分块（chunk）预测，避免一次性构造超大 batch；
    - 逐历元处理，当前历元角点预测用完即释放。
"""

import numpy as np
import torch
import pandas as pd

# 复用 test.py 中已有的批量构造与推理逻辑，避免重复造轮子
from scripts.test import collate_inference_batch, predict_batch, _extract_epoch_arrays
from data.dataset import map_system_id_to_index


# ======================================================================
# 1. 格网轴与格网点构造
# ======================================================================

def build_product_grid_axes(product_cfg: dict, coord_norm=None) -> dict:
    """
    根据配置生成 lat / lon / az / el 四个坐标轴（1D 递增数组）。

    边界处理：包含终点（np.linspace 含 endpoint），与项目经纬度范围处理一致。
    缺省值：lat/lon 若未配置则从 coord_norm 的 min/max 推断；az∈[0,360]、el∈[0,90]。

    Args:
        product_cfg: inference.product_eval 配置子节
        coord_norm:  CoordNormalizer（用于推断 lat/lon 缺省范围）

    Returns:
        axes: dict，键 lat/lon/az/el，值为各维 1D np.ndarray（float64）
    """
    def _axis(vmin, vmax, vres):
        # vres 为分辨率（步长）。点数 = round((max-min)/res) + 1，含终点。
        n = int(round((vmax - vmin) / vres)) + 1
        n = max(n, 2)  # 至少 2 个点才能构成 cell
        return np.linspace(vmin, vmax, n, dtype=np.float64)

    # lat/lon 缺省范围：优先配置，否则用 coord_norm 的训练范围
    # 注意：YAML 中显式写 null 时 .get 会返回 None，需用 _or_default 回退
    def _or_default(key, default):
        v = product_cfg.get(key, default)
        return default if v is None else v

    lat_min = _or_default("lat_min", coord_norm.lat_min if coord_norm else -90.0)
    lat_max = _or_default("lat_max", coord_norm.lat_max if coord_norm else 90.0)
    lon_min = _or_default("lon_min", coord_norm.lon_min if coord_norm else -180.0)
    lon_max = _or_default("lon_max", coord_norm.lon_max if coord_norm else 180.0)
    az_min  = _or_default("az_min", 0.0)
    az_max  = _or_default("az_max", 360.0)
    el_min  = _or_default("el_min", 0.0)
    el_max  = _or_default("el_max", 90.0)

    lat_res = _or_default("lat_res", 2.0)
    lon_res = _or_default("lon_res", 2.0)
    az_res  = _or_default("az_res", 30.0)
    el_res  = _or_default("el_res", 15.0)

    return {
        "lat": _axis(lat_min, lat_max, lat_res),
        "lon": _axis(lon_min, lon_max, lon_res),
        "az":  _axis(az_min, az_max, az_res),
        "el":  _axis(el_min, el_max, el_res),
    }


def build_product_grid_points(axes: dict) -> np.ndarray:
    """
    由四个坐标轴生成完整 4D 格网点（笛卡尔积）。

    注意：仅用于统计 theoretical_total_product_grid_points 或非 corner-only
    模式。corner-only 模式下不调用本函数（避免构造超大数组）。

    Args:
        axes: build_product_grid_axes 的返回值

    Returns:
        points: [P, 4]  每行 (lat, lon, az, el)
    """
    lat, lon, az, el = axes["lat"], axes["lon"], axes["az"], axes["el"]
    mesh = np.meshgrid(lat, lon, az, el, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=-1)  # [P, 4]
    return pts.astype(np.float64)


# ======================================================================
# 2. 4D cell 定位与角点提取
# ======================================================================

def _locate_bin(value: float, axis: np.ndarray):
    """
    在递增 axis 上定位 value 所在的 bin（区间 [axis[i], axis[i+1]]）。

    返回下界索引 i（0 <= i <= len(axis)-2）。越界返回 None。
    边界处理：value 落在格网边界上时，选包含它的相邻 cell（用 searchsorted
    的 'right' 语义后回退一格，保证右端点归入最后一个 cell）。

    Args:
        value: 标量
        axis:  递增 1D 数组
    Returns:
        i 或 None（越界，不外推）
    """
    if value < axis[0] or value > axis[-1]:
        return None  # 越界，不外推
    # searchsorted 找插入位，减 1 得 bin 下界索引
    i = int(np.searchsorted(axis, value, side="right")) - 1
    # 右端点 value == axis[-1] 时 i = len-1，回退到最后一个 cell
    if i >= len(axis) - 1:
        i = len(axis) - 2
    if i < 0:
        i = 0
    return i


def locate_4d_grid_cell(lat, lon, az, el, axes: dict):
    """
    根据 val IPP 的 (lat, lon, az, el) 定位其所在 4D cell。

    Args:
        lat, lon, az, el: 标量
        axes: 格网轴 dict
    Returns:
        cell_idx: (i_lat, i_lon, i_az, i_el) 四元组，或 None（任一维越界）
    """
    i_lat = _locate_bin(lat, axes["lat"])
    i_lon = _locate_bin(lon, axes["lon"])
    i_az  = _locate_bin(az,  axes["az"])
    i_el  = _locate_bin(el,  axes["el"])
    if None in (i_lat, i_lon, i_az, i_el):
        return None  # out_of_grid
    return (i_lat, i_lon, i_az, i_el)


def get_4d_cell_corners(cell_idx, axes: dict) -> np.ndarray:
    """
    获取 4D cell 的全部 16 个角点坐标。

    Args:
        cell_idx: (i_lat, i_lon, i_az, i_el)
        axes:     格网轴 dict
    Returns:
        corners: [16, 4]  每行 (lat, lon, az, el)
    """
    i_lat, i_lon, i_az, i_el = cell_idx
    lat_pair = axes["lat"][i_lat:i_lat + 2]  # [2]
    lon_pair = axes["lon"][i_lon:i_lon + 2]
    az_pair  = axes["az"][i_az:i_az + 2]
    el_pair  = axes["el"][i_el:i_el + 2]

    corners = []
    for la in lat_pair:
        for lo in lon_pair:
            for a in az_pair:
                for e in el_pair:
                    corners.append((la, lo, a, e))
    return np.asarray(corners, dtype=np.float64)  # [16, 4]


# ======================================================================
# 3. 角点收集去重（corner-only 优化）+ IDW 插值
# ======================================================================

def collect_required_product_corners(val_pts: np.ndarray, axes: dict):
    """
    corner-only 优化核心：针对当前历元的 val IPP 点，收集评估所需的 4D 角点
    并去重，只预测这些角点而非整个格网，显著降低显存/内存占用。

    Args:
        val_pts: [V, 4]  val IPP 点 (lat, lon, az, el)
        axes:    格网轴 dict
    Returns:
        unique_corners: [U, 4]   去重后的角点坐标
        val_cell_corner_idx: list（长度 V），每元素为该 val 点 16 角点在
                             unique_corners 中的行索引数组 [16]；越界点为 None
        n_skipped: int  越界（out_of_grid）val 点数
    """
    corner_key_to_row = {}   # (la,lo,az,el) -> row index in unique list
    unique_list = []
    val_cell_corner_idx = []
    n_skipped = 0

    for v in val_pts:
        cell = locate_4d_grid_cell(v[0], v[1], v[2], v[3], axes)
        if cell is None:
            val_cell_corner_idx.append(None)
            n_skipped += 1
            continue
        corners = get_4d_cell_corners(cell, axes)  # [16, 4]
        idx16 = np.empty(16, dtype=np.int64)
        for j, c in enumerate(corners):
            key = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            row = corner_key_to_row.get(key)
            if row is None:
                row = len(unique_list)
                corner_key_to_row[key] = row
                unique_list.append(c)
            idx16[j] = row
        val_cell_corner_idx.append(idx16)

    unique_corners = (np.asarray(unique_list, dtype=np.float64)
                      if unique_list else np.zeros((0, 4), dtype=np.float64))
    return unique_corners, val_cell_corner_idx, n_skipped


def idw_from_4d_corners(q: np.ndarray, corners: np.ndarray,
                        corner_stec: np.ndarray, p: float = 2.0,
                        eps: float = 1e-8) -> float:
    """
    用 4D cell 的 16 个角点 STEC 对 val IPP 点 q 做 IDW 插值。

    距离在「归一化后」的 4D 空间计算（q 与 corners 都应已归一化），
    避免 lat/lon/az/el 量纲差异导致某一维主导。

    若 q 与某角点距离为 0，直接返回该角点 STEC（防除零）。

    Args:
        q:           [4]      归一化后的 val 点 (lat,lon,az,el)
        corners:     [16, 4]  归一化后的角点坐标
        corner_stec: [16]     角点预测 STEC（原始 TECU 单位）
        p:           IDW 幂次
        eps:         数值稳定小量
    Returns:
        pred: 标量 STEC
    """
    diff = corners - q[None, :]           # [16, 4]
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # [16]

    # 命中角点：距离为 0（或极小）直接返回该角点值，避免除零
    hit = np.where(dist < eps)[0]
    if hit.size > 0:
        return float(corner_stec[hit[0]])

    w = 1.0 / np.power(dist + eps, p)     # [16]
    return float((w * corner_stec).sum() / w.sum())


# ======================================================================
# 4. 分块角点预测（corner-only，每历元独立）
# ======================================================================

def predict_product_grid_chunked(
    model, sde, unique_corners: np.ndarray,
    ctx_arrays: tuple, grid_sys_idx: int,
    coord_norm, stec_norm, angle_norm, device,
    chunk_size: int = 4096, verbose: bool = False,
) -> np.ndarray:
    """
    分块调用模型，对去重后的格网角点预测 STEC（原始 TECU 单位）。

    角点作 target、model_stations 作 context。复用 test.py 的
    collate_inference_batch + predict_batch，保证与 direct 推理完全一致的
    归一化 / batch 构造 / 反向 SDE / 反归一化流程。

    Args:
        model, sde:   已加载模型与 SDE
        unique_corners: [U, 4]  去重角点 (lat,lon,az,el)，原始单位
        ctx_arrays:   context（model_stations）的数组元组：
                      (ctx_lats, ctx_lons, ctx_az, ctx_el, ctx_stec,
                       ctx_sys_ids, ctx_sat_ids, ctx_stations)
        grid_sys_idx: 格网点统一使用的 system 索引（该历元 context 的众数系统）
        coord_norm/stec_norm/angle_norm: 归一化器
        device:       torch.device
        chunk_size:   每块最多预测多少角点（防显存溢出）
        verbose:      是否打印反向扩散进度
    Returns:
        corner_stec: [U]  角点预测 STEC（原始 TECU 单位）
    """
    (ctx_lats, ctx_lons, ctx_az, ctx_el, ctx_stec,
     ctx_sys_ids, ctx_sat_ids, ctx_stations) = ctx_arrays

    U = unique_corners.shape[0]
    if U == 0:
        return np.zeros((0,), dtype=np.float32)

    corner_stec = np.empty(U, dtype=np.float32)
    n_chunks = (U + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, U)
        chunk = unique_corners[s:e]  # [m, 4]
        m = chunk.shape[0]

        # 角点作 target。tgt_stec 为 dummy（label 不参与预测，predict_batch
        # 内部会用噪声覆盖 target 位置；仅用于占位以复用 collate）。
        tgt_lats = chunk[:, 0].astype(np.float32)
        tgt_lons = chunk[:, 1].astype(np.float32)
        tgt_az   = chunk[:, 2].astype(np.float32)
        tgt_el   = chunk[:, 3].astype(np.float32)
        tgt_stec = np.zeros(m, dtype=np.float32)                 # dummy label
        tgt_sys  = np.full(m, grid_sys_idx, dtype=np.int64)      # 统一系统索引
        tgt_sat  = np.full(m, -1, dtype=np.int64)                # dummy 卫星号
        tgt_stations = ["__grid__"] * m                          # dummy 站名

        epoch_data = (
            ctx_lats, ctx_lons, ctx_az, ctx_el, ctx_stec,
            ctx_sys_ids, ctx_sat_ids, ctx_stations,
            tgt_lats, tgt_lons, tgt_az, tgt_el, tgt_stec,
            tgt_sys, tgt_sat, tgt_stations,
        )

        batch_tensors, meta_list = collate_inference_batch(
            [epoch_data], coord_norm, stec_norm, angle_norm, device)
        result_dfs = predict_batch(
            model, sde, batch_tensors, meta_list, stec_norm, device, verbose)

        # predict_batch 的 target 顺序与输入一致，pred_stec 即角点预测
        corner_stec[s:e] = result_dfs[0]["pred_stec"].values.astype(np.float32)

        # 及时释放
        del batch_tensors, meta_list, result_dfs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return corner_stec


def _normalize_4d(pts: np.ndarray, coord_norm, angle_norm) -> np.ndarray:
    """
    将 [N,4] 的 (lat,lon,az,el) 用与模型输入一致的归一化器映射到归一化空间，
    供 IDW 距离计算使用（避免量纲差异）。

    coord_norm.transform 返回 [N,2]=(lat_n,lon_n)；angle_norm.transform 同理。
    """
    ll = coord_norm.transform(pts[:, 0], pts[:, 1])   # [N,2]
    ae = angle_norm.transform(pts[:, 2], pts[:, 3])   # [N,2]
    return np.concatenate([ll, ae], axis=-1).astype(np.float64)  # [N,4]


# ======================================================================
# 5. product evaluation 主入口
# ======================================================================

def run_product_evaluation(
    model, sde, valid_epochs, model_files, val_files,
    system_ascii_code, product_cfg,
    coord_norm, stec_norm, angle_norm, device, verbose=False,
):
    """
    product / grid-product 评估主入口。

    逐历元执行：
      1. 提取 context(model_stations) 与 val(val_stations) 数组；
      2. 为该历元构建 4D 格网轴（不区分卫星）；
      3. corner-only：收集 val 点所在 cell 的去重角点；
      4. 模型分块预测这些角点 STEC（context=model_stations）；
      5. 每个 val 点用其 16 角点 IDW 插值得到 product_pred_STEC；
      6. 聚合 val 级预测，越界点跳过并计数。

    Returns:
        val_results_df: DataFrame（val 级：label / product_pred / 误差 / 元信息）
        stats: dict（格网与统计量）
    """
    axes = build_product_grid_axes(product_cfg, coord_norm)
    idw_power  = float(product_cfg.get("idw_power", 2.0))
    chunk_size = int(product_cfg.get("chunk_size", 4096))
    corner_only = bool(product_cfg.get("corner_only", True))

    lat_n, lon_n = len(axes["lat"]), len(axes["lon"])
    az_n,  el_n  = len(axes["az"]),  len(axes["el"])
    theoretical_total = lat_n * lon_n * az_n * el_n

    all_rows = []
    total_val = 0
    valid_val = 0
    skipped = 0
    total_corners_evaluated = 0

    n_epochs = len(valid_epochs)
    for ei, stem in enumerate(valid_epochs):
        model_df = pd.read_csv(model_files[stem])
        val_df   = pd.read_csv(val_files[stem])
        try:
            arrays = _extract_epoch_arrays(model_df, val_df, system_ascii_code)
        except Exception as e:
            print(f"    [Warning] 历元 {stem} 数据提取失败：{e}")
            continue

        (ctx_lats, ctx_lons, ctx_az, ctx_el, ctx_stec,
         ctx_sys_ids, ctx_sat_ids, ctx_stations,
         tgt_lats, tgt_lons, tgt_az, tgt_el, tgt_stec,
         tgt_sys_ids, tgt_sat_ids, tgt_stations) = arrays

        if len(tgt_lats) == 0 or len(ctx_lats) == 0:
            continue

        # 格网点统一 system 索引：取 context 众数系统（单系统场景天然一致）
        if len(ctx_sys_ids) > 0:
            grid_sys_idx = int(np.bincount(ctx_sys_ids.astype(np.int64)).argmax())
        else:
            grid_sys_idx = 1

        ctx_arrays = (ctx_lats, ctx_lons, ctx_az, ctx_el, ctx_stec,
                      ctx_sys_ids, ctx_sat_ids, ctx_stations)

        # val 点 4D 坐标（原始单位）
        val_pts = np.stack([tgt_lats, tgt_lons, tgt_az, tgt_el], axis=-1).astype(np.float64)

        # corner-only：收集去重角点 + 每个 val 点的 16 角点索引
        unique_corners, val_corner_idx, n_skip = collect_required_product_corners(val_pts, axes)
        skipped += n_skip
        total_corners_evaluated += unique_corners.shape[0]

        # 模型预测去重角点 STEC（原始 TECU）
        corner_stec = predict_product_grid_chunked(
            model, sde, unique_corners, ctx_arrays, grid_sys_idx,
            coord_norm, stec_norm, angle_norm, device,
            chunk_size=chunk_size, verbose=verbose,
        )

        # 归一化角点与 val 点（IDW 距离用归一化空间）
        if unique_corners.shape[0] > 0:
            corners_norm = _normalize_4d(unique_corners, coord_norm, angle_norm)
        val_pts_norm = _normalize_4d(val_pts, coord_norm, angle_norm)

        # 逐 val 点 IDW
        for vi in range(len(tgt_lats)):
            total_val += 1
            idx16 = val_corner_idx[vi]
            if idx16 is None:
                continue  # out_of_grid，跳过（不外推）
            q = val_pts_norm[vi]                    # [4] 归一化
            c16 = corners_norm[idx16]               # [16,4]
            s16 = corner_stec[idx16]                # [16]
            pred = idw_from_4d_corners(q, c16, s16, p=idw_power)
            true = float(tgt_stec[vi])
            valid_val += 1
            all_rows.append({
                "epoch_time":    stem,
                "station_name":  tgt_stations[vi],
                "ipp_latitude":  float(tgt_lats[vi]),
                "ipp_longitude": float(tgt_lons[vi]),
                "azimuth_deg":   float(tgt_az[vi]),
                "elevation_deg": float(tgt_el[vi]),
                "system_id":     int(tgt_sys_ids[vi]),
                "satellite_id":  int(tgt_sat_ids[vi]),
                "true_stec":     true,
                "pred_stec":     pred,
                "abs_error":     abs(pred - true),
            })

        # 及时释放当前历元角点预测
        del unique_corners, corner_stec
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (ei + 1) % 10 == 0 or (ei + 1) == n_epochs:
            print(f"  [product {ei+1}/{n_epochs}] 已处理历元，valid_val={valid_val} skipped={skipped}")

    val_results_df = pd.DataFrame(all_rows)
    stats = {
        "lat_grid_count": lat_n,
        "lon_grid_count": lon_n,
        "az_grid_count":  az_n,
        "el_grid_count":  el_n,
        "theoretical_total_product_grid_points": theoretical_total,
        "evaluated_corner_points": total_corners_evaluated,
        "total_val_points": total_val,
        "valid_val_points": valid_val,
        "skipped_out_of_grid_points": skipped,
        "chunk_size": chunk_size,
        "idw_power": idw_power,
        "whether_corner_only_optimization_enabled": corner_only,
    }
    return val_results_df, stats
