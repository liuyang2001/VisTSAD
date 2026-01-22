# data_utils.py
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from config import NUM_VARS, WINDOW_SIZE, ROLLING_VAR_WINDOW

# 互信息直方图 bin 数
MI_N_BINS = 16


# ============================================================
# 1. 读取 Excel
# ============================================================
def load_excel_with_boundary(path: Path):
    """
    约定：
      - 前 NUM_VARS 列：特征
      - 倒数第 2 列：Label (0 正常 / 1 异常)
      - 倒数第 1 列：Segment_Boundary (0/1，1 表示某段的最后一个时间步)

    返回：
      data: (D, T)
      labels: (T,)
      seg_boundary: (T,)
    """
    df = pd.read_excel(path)

    feature_cols = df.columns[:NUM_VARS]
    label_col = df.columns[-2]
    seg_col = df.columns[-1]

    data = df[feature_cols].to_numpy(dtype=float).T  # (T,D)->(D,T)
    labels = df[label_col].to_numpy(dtype=int)
    seg_boundary = df[seg_col].to_numpy(dtype=int)

    return data, labels, seg_boundary


# ============================================================
# 2. 滑窗（带历史，不跨段）
# ============================================================
def sliding_windows_with_boundary(data, seg_boundary, window_size, stride):
    """
    输入：
      data: (D, T)
      seg_boundary: (T,), 1 表示该时间步为某段最后一个点

    返回：
      windows: List[(D, L_ext)]，每个窗口带有 K-1 步历史
      idx_pairs: List[(start, end)]  窗口核心区间的起止索引（在原时间序列上）
    """
    D, T = data.shape
    windows = []
    idx_pairs = []

    K = ROLLING_VAR_WINDOW

    seg_start = 0
    for t in range(T):
        if seg_boundary[t] == 1 or t == T - 1:
            seg_end = t
            seg_len = seg_end - seg_start + 1
            min_len = window_size + (K - 1)
            if seg_len < min_len:
                seg_start = t + 1
                continue

            first_win_start = seg_start + (K - 1)
            win_start = first_win_start
            while win_start + window_size - 1 <= seg_end:
                win_end = win_start + window_size - 1
                hist_start = win_start - (K - 1)
                hist_end = win_end
                w_ext = data[:, hist_start : hist_end + 1]
                windows.append(w_ext)
                idx_pairs.append((win_start, win_end))
                win_start += stride
            seg_start = t + 1

    return windows, idx_pairs

# ============================================================
# 4. RGB 图构建（使用 Min-Max）
# ============================================================
def window_to_rgb(
    window_ext: np.ndarray,
    var_order: List[int],
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    diff_min_vals: np.ndarray,
    diff_max_vals: np.ndarray,
    var_min_vals: np.ndarray,
    var_max_vals: np.ndarray,
) -> np.ndarray:
    """
    使用全局 Min-Max 归一化（训练 + 测试）
    输出 RGB (D, W, 3)，D = 变量数（按 var_order 排序），W = WINDOW_SIZE
    通道含义：
      R: 原始值
      G: 一阶差分
      B: rolling 方差
    """
    # 按排序后的顺序
    w_full = window_ext[var_order, :]  # (D, L_ext)

    min_v = min_vals[var_order]
    max_v = max_vals[var_order]
    diff_min = diff_min_vals[var_order]
    diff_max = diff_max_vals[var_order]
    var_min = var_min_vals[var_order]
    var_max = var_max_vals[var_order]

    # ----- R 通道：原始值 -----
    w_core = w_full[:, -WINDOW_SIZE:]  # (D, W)
    ch1 = (w_core - min_v[:, None]) / (max_v[:, None] - min_v[:, None])
    ch1 = np.clip(ch1, 0, 1)

    # ----- G 通道：差分 -----
    diff_full = np.diff(w_full, axis=1)
    diff_full = np.concatenate(
        [np.zeros((w_full.shape[0], 1)), diff_full], axis=1
    )
    diff_core = diff_full[:, -WINDOW_SIZE:]

    ch2 = (diff_core - diff_min[:, None]) / (diff_max[:, None] - diff_min[:, None])
    ch2 = np.clip(ch2, 0, 1)

    # ----- B 通道：rolling variance -----
    D_, L_ext = w_full.shape
    var_full = np.zeros_like(w_full)
    for i in range(D_):
        s = pd.Series(w_full[i])
        var_full[i] = s.rolling(
            ROLLING_VAR_WINDOW, min_periods=ROLLING_VAR_WINDOW
        ).var().values
    var_core = var_full[:, -WINDOW_SIZE:]
    var_core = np.nan_to_num(var_core, nan=0.0)

    ch3 = (var_core - var_min[:, None]) / (var_max[:, None] - var_min[:, None])
    ch3 = np.clip(ch3, 0, 1)

    # ----- Stack (= RGB) -----
    rgb = np.stack([ch1, ch2, ch3], axis=-1)  # (D, W, 3)
    # rgb = (rgb * 255).astype(np.uint8)

    return rgb


# ============================================================
# 5. 正常库中找最相似窗口（相关矩阵 / RGB）
# ============================================================


def find_most_similar_rgb_window(
    rgb_db: List[np.ndarray],
    rgb_test: np.ndarray,
) -> Tuple[int, float]:
    """
    在 RGB 库中找最相似窗口（MSE）。
    rgb_db: List[(D,W,3)]
    rgb_test: (D,W,3)
    """
    best_idx = -1
    best_dist = float("inf")

    rgb_test_f = rgb_test.astype(np.float32)

    for i, rgb in enumerate(rgb_db):
        if rgb.shape != rgb_test.shape:
            continue
        diff = rgb - rgb_test_f
        dist = np.mean(diff * diff)  # MSE

        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx, best_dist


# ============================================================
# 6. 全局 Min-Max 统计（值 / 差分 / rolling variance）
# ============================================================
def compute_minmax_stats(data: np.ndarray):
    """
    根据 data (D, T) 计算每个变量的 (min, max)。
    返回：
        min_vals: (D,)
        max_vals: (D,)
    """
    min_vals = data.min(axis=1)
    max_vals = data.max(axis=1)

    # 避免除零问题
    eps = 1e-6
    max_vals = np.where(max_vals - min_vals < eps, min_vals + eps, max_vals)

    return min_vals, max_vals


def compute_diff_var_minmax_stats(
    data: np.ndarray,
    seg_boundary: np.ndarray,
):
    """
    根据 data (D, T) 计算：
        - 每个变量的一阶差分的 min/max
        - 每个变量 rolling variance 的 min/max（不跨段）

    返回：
        diff_min_vals: (D,)
        diff_max_vals: (D,)
        var_min_vals:  (D,)
        var_max_vals:  (D,)
    """
    D, T = data.shape
    K = ROLLING_VAR_WINDOW

    # ---------- 一阶差分 ----------
    diff = np.diff(data, axis=1)  # (D, T-1)
    # 避免跨段：seg_boundary[t] == 1 表示 t 是段尾，对应 diff[:, t] 不合法
    invalid = seg_boundary[:-1] == 1
    diff[:, invalid] = np.nan

    diff_min_vals = np.nanmin(diff, axis=1)
    diff_max_vals = np.nanmax(diff, axis=1)

    eps = 1e-6
    diff_max_vals = np.where(
        diff_max_vals - diff_min_vals < eps,
        diff_min_vals + eps,
        diff_max_vals,
    )

    # ---------- rolling variance（不跨段） ----------
    var_list = []

    for i in range(D):
        v_all = np.full(T, np.nan, dtype=float)
        seg_start = 0
        for t in range(T):
            if seg_boundary[t] == 1 or t == T - 1:
                seg_end = t
                seg_len = seg_end - seg_start + 1
                if seg_len >= K:
                    s = pd.Series(data[i, seg_start : seg_end + 1])
                    v_seg = s.rolling(K, min_periods=K).var().values
                    v_all[seg_start : seg_end + 1] = v_seg
                seg_start = t + 1
        var_list.append(v_all)

    var_arr = np.stack(var_list, axis=0)  # (D, T)

    var_min_vals = np.nanmin(var_arr, axis=1)
    var_max_vals = np.nanmax(var_arr, axis=1)

    var_max_vals = np.where(
        var_max_vals - var_min_vals < eps,
        var_min_vals + eps,
        var_max_vals,
    )

    return diff_min_vals, diff_max_vals, var_min_vals, var_max_vals


# ============================================================
# 7. 欧几里得距离矩阵（值 / 差分 / rolling var）
# ============================================================
def pairwise_rms_distance(mat: np.ndarray) -> np.ndarray:
    """
    mat: (D, L) 矩阵，每一行是一个变量在该窗口上的时间序列。
    返回:
        dist: (D, D)，dist[i, j] = sqrt( mean_t ( mat[i,t] - mat[j,t] )^2 )
    即每一对变量序列之间的“均方根欧氏距离”，越大表示两个变量在该窗口上的动态越不相似。
    """
    diff = mat[:, None, :] - mat[None, :, :]  # (D, D, L)
    dist = np.sqrt(np.mean(diff * diff, axis=2))  # (D, D)
    return dist


def compute_mean_rel_matrices(
    window_ext: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输入:
        window_ext: (D, L_ext)，已经按 var_order 排好序
    输出:
        M_value, M_diff, M_var  均为 (D, D) 矩阵
        元素定义为：对应变量对在该窗口上的时间序列的欧几里得距离（RMS）。
    """
    D, _ = window_ext.shape

    # 1) 值通道
    M_value = pairwise_rms_distance(window_ext)

    # 2) 差分通道
    diff = np.diff(window_ext, axis=1)
    diff = np.concatenate([np.zeros((D, 1)), diff], axis=1)  # 对齐长度
    M_diff = pairwise_rms_distance(diff)

    # 3) rolling variance 通道
    var_full = np.zeros_like(window_ext)
    for i in range(D):
        s = pd.Series(window_ext[i])
        var_full[i] = s.rolling(
            ROLLING_VAR_WINDOW,
            min_periods=ROLLING_VAR_WINDOW,
        ).var().values
    var_full = np.nan_to_num(var_full, nan=0.0)
    M_var = pairwise_rms_distance(var_full)

    return M_value, M_diff, M_var


# ============================================================
# 8. 互信息矩阵（值 / 差分 / rolling var）
# ============================================================
def mutual_info_1d(x: np.ndarray, y: np.ndarray, n_bins: int = MI_N_BINS) -> float:
    """
    用直方图估计两个一维连续随机变量的互信息：
        MI(X;Y) = sum_{i,j} p(x_i, y_j) log( p(x_i, y_j) / (p(x_i) p(y_j)) )

    参数:
        x, y : (L,) 一维数组，对应两个变量在窗口上的时间序列
    返回:
        mi : 标量，互信息（单位为 nats）
    """
    # 去掉 NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return 0.0

    # 2D 直方图
    c_xy, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    total = c_xy.sum()
    if total <= 0:
        return 0.0

    p_xy = c_xy / total  # 联合分布
    p_x = p_xy.sum(axis=1)  # (n_bins,)
    p_y = p_xy.sum(axis=0)  # (n_bins,)

    # 避免 log(0)
    px_py = p_x[:, None] * p_y[None, :]
    mask_nonzero = (p_xy > 0) & (px_py > 0)

    mi = np.sum(p_xy[mask_nonzero] * np.log(p_xy[mask_nonzero] / px_py[mask_nonzero]))
    return float(mi)


def pairwise_mi_matrix(mat: np.ndarray, n_bins: int = MI_N_BINS) -> np.ndarray:
    """
    mat: (D, L)，D 个变量的时间序列
    返回:
        M: (D, D)，M[i,j] = MI( mat[i,:], mat[j,:] )
    """
    D, L = mat.shape
    M = np.zeros((D, D), dtype=float)

    for i in range(D):
        for j in range(D):
            M[i, j] = mutual_info_1d(mat[i], mat[j], n_bins=n_bins)

    return M


def compute_mi_matrices(
    window_ext: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输入:
        window_ext: (D, L_ext)，已经按 var_order 排好序
    输出:
        M_val, M_diff, M_var  均为 (D, D) 矩阵
        元素为对应变量对在该通道下的互信息 MI。
    """
    D, L_ext = window_ext.shape

    # 1) 值通道
    M_val = pairwise_mi_matrix(window_ext)

    # 2) 差分通道
    diff = np.diff(window_ext, axis=1)
    diff = np.concatenate([np.zeros((D, 1)), diff], axis=1)
    M_diff = pairwise_mi_matrix(diff)

    # 3) rolling variance 通道
    var_full = np.zeros_like(window_ext)
    for i in range(D):
        s = pd.Series(window_ext[i])
        var_full[i] = s.rolling(
            ROLLING_VAR_WINDOW,
            min_periods=ROLLING_VAR_WINDOW,
        ).var().values
    var_full = np.nan_to_num(var_full, nan=0.0)
    M_var = pairwise_mi_matrix(var_full)

    return M_val, M_diff, M_var


# ============================================================
# 9. 基于 MI 的变量排序（用于 main.build_priors）
# ============================================================
def mi_based_order(data: np.ndarray) -> List[int]:
    """
    基于互信息做层次聚类的变量排序。

    参数：
        data: (D, T)，全训练数据（或拼接后的大数据）
    返回：
        order: List[int]，变量的新顺序索引
    """
    D, T = data.shape
    # 先算 MI 矩阵（非负）
    M_mi = pairwise_mi_matrix(data)  # (D, D)

    max_mi = float(M_mi.max())
    if max_mi <= 0:
        # 极端情况（全部 0），直接用原始顺序
        return list(range(D))

    # 将 MI 归一化到 [0,1]，再转成距离（越大越不相似）
    sim = M_mi / max_mi  # [0,1]
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    return order.tolist()
