# generate_images.py
from pathlib import Path
from typing import List, Dict, Any
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    TRAIN_PATH,
    OUTPUT_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    NUM_VARS,
    TRAIN_LIB_DIR,
    EU1,
    EU2,
    MI1,
    MI2
)

from data_utils import (
    load_excel_with_boundary,
    sliding_windows_with_boundary,
    compute_minmax_stats,
    compute_diff_var_minmax_stats,
    window_to_rgb,
    find_most_similar_rgb_window,
    compute_mean_rel_matrices,
    compute_mi_matrices,
    mi_based_order
)

from plotting import save_ts_heatmap,save_corr_heatmap
def build_priors(
    train_data: np.ndarray,
    train_seg: np.ndarray,
    feature_names: List[str],
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    diff_min_vals: np.ndarray,
    diff_max_vals: np.ndarray,
    var_min_vals: np.ndarray,
    var_max_vals: np.ndarray,
    test_windows_ext: List[np.ndarray],
):
    print("[INFO] Building priors from train_data ...")

    windows_ext, _ = sliding_windows_with_boundary(
        train_data, train_seg, WINDOW_SIZE, stride=WINDOW_STRIDE
    )
    n_train_win = len(windows_ext)
    print(f"[INFO] #train windows (extended) = {n_train_win}")

    var_order = mi_based_order(train_data)
    ordered_names = [feature_names[i] for i in var_order]
    print("[INFO] Variable order after MI-based hierarchical clustering:")
    for idx, (orig_idx, name) in enumerate(zip(var_order, ordered_names)):
        print(f"  order {idx:02d} <- original {orig_idx:02d} : {name}")

    train_windows_ord = [w_ext[var_order, :] for w_ext in windows_ext]
    D = train_data.shape[0]

    rgb_db = []
    for _, w_ext in enumerate(windows_ext):
        rgb = window_to_rgb(
            w_ext,
            var_order,
            min_vals,
            max_vals,
            diff_min_vals,
            diff_max_vals,
            var_min_vals,
            var_max_vals,
        )
        rgb_db.append(rgb.astype(np.float32))

    print(f"[INFO] Built train RGB library (in memory), len={len(rgb_db)}")

    # EU global min/max
    print("[INFO] Computing global min/max for Euclidean matrices on train+test windows ...")
    eu_val_min = np.full((D, D), np.inf)
    eu_val_max = np.full((D, D), -np.inf)
    eu_diff_min = np.full((D, D), np.inf)
    eu_diff_max = np.full((D, D), -np.inf)
    eu_var_min = np.full((D, D), np.inf)
    eu_var_max = np.full((D, D), -np.inf)

    for w_ord in train_windows_ord:
        Mv_eu, Md_eu, Mvar_eu = compute_mean_rel_matrices(w_ord)
        eu_val_min = np.minimum(eu_val_min, Mv_eu)
        eu_val_max = np.maximum(eu_val_max, Mv_eu)
        eu_diff_min = np.minimum(eu_diff_min, Md_eu)
        eu_diff_max = np.maximum(eu_diff_max, Md_eu)
        eu_var_min = np.minimum(eu_var_min, Mvar_eu)
        eu_var_max = np.maximum(eu_var_max, Mvar_eu)

    for w_ext in test_windows_ext:
        if w_ext.shape[1] < WINDOW_SIZE:
            continue
        w_ord = w_ext[var_order, :]
        Mv_eu, Md_eu, Mvar_eu = compute_mean_rel_matrices(w_ord)
        eu_val_min = np.minimum(eu_val_min, Mv_eu)
        eu_val_max = np.maximum(eu_val_max, Mv_eu)
        eu_diff_min = np.minimum(eu_diff_min, Md_eu)
        eu_diff_max = np.maximum(eu_diff_max, Md_eu)
        eu_var_min = np.minimum(eu_var_min, Mvar_eu)
        eu_var_max = np.maximum(eu_var_max, Mvar_eu)

    # MI global min/max
    print("[INFO] Computing global min/max for MI matrices on train+test windows ...")
    mi_val_min = np.full((D, D), np.inf)
    mi_val_max = np.full((D, D), -np.inf)
    mi_diff_min = np.full((D, D), np.inf)
    mi_diff_max = np.full((D, D), -np.inf)
    mi_var_min = np.full((D, D), np.inf)
    mi_var_max = np.full((D, D), -np.inf)

    for w_ord in train_windows_ord:
        Mv_mi, Md_mi, Mvar_mi = compute_mi_matrices(w_ord)
        mi_val_min = np.minimum(mi_val_min, Mv_mi)
        mi_val_max = np.maximum(mi_val_max, Mv_mi)
        mi_diff_min = np.minimum(mi_diff_min, Md_mi)
        mi_diff_max = np.maximum(mi_diff_max, Md_mi)
        mi_var_min = np.minimum(mi_var_min, Mvar_mi)
        mi_var_max = np.maximum(mi_var_max, Mvar_mi)

    for w_ext in test_windows_ext:
        if w_ext.shape[1] < WINDOW_SIZE:
            continue
        w_ord = w_ext[var_order, :]
        Mv_mi, Md_mi, Mvar_mi = compute_mi_matrices(w_ord)
        mi_val_min = np.minimum(mi_val_min, Mv_mi)
        mi_val_max = np.maximum(mi_val_max, Mv_mi)
        mi_diff_min = np.minimum(mi_diff_min, Md_mi)
        mi_diff_max = np.maximum(mi_diff_max, Md_mi)
        mi_var_min = np.minimum(mi_var_min, Mvar_mi)
        mi_var_max = np.maximum(mi_var_max, Mvar_mi)

    print("[INFO] Finished building priors.")

    return (
        var_order,
        min_vals,
        max_vals,
        diff_min_vals,
        diff_max_vals,
        var_min_vals,
        var_max_vals,
        feature_names,
        rgb_db,
        eu_val_min, eu_val_max, eu_diff_min, eu_diff_max, eu_var_min, eu_var_max,
        mi_val_min, mi_val_max, mi_diff_min, mi_diff_max, mi_var_min, mi_var_max,
    )
def soft_threshold_diff(M: np.ndarray, t1: float = 0.3, t2: float = 0.5) -> np.ndarray:
    A = np.abs(M)
    sign = np.sign(M)

    out = np.zeros_like(M, dtype=float)

    mid_mask = (A >= t1) & (A < t2)
    out[mid_mask] = (A[mid_mask] - t1) / (t2 - t1)

    high_mask = A >= t2
    out[high_mask] = 1.0

    return out * sign

def make_diff_channel(value: np.ndarray) -> np.ndarray:
    # value: (D,W)
    out = np.zeros_like(value)
    out[:, 1:] = value[:, 1:] - value[:, :-1]
    return out


def make_rolling_var_channel(value: np.ndarray, win: int = 7) -> np.ndarray:
    # value: (D,W)
    D, W = value.shape
    out = np.zeros((D, W), dtype=float)
    for t in range(W):
        s = max(0, t - win + 1)
        seg = value[:, s : t + 1]
        out[:, t] = np.var(seg, axis=1)
    return out


def normalize_per_var(X: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    # X: (D,W), vmin/vmax: (D,)
    rng = (vmax - vmin).astype(float)
    rng[rng < 1e-9] = 1.0
    Y = (X - vmin[:, None]) / rng[:, None]
    return np.clip(Y, 0.0, 1.0)


# ============================================================
# priors cache
# ============================================================
TRAIN_PRIORS_PATH = TRAIN_LIB_DIR / "train_priors.npz"

PARAM_CONFIGS = [
    {"EU_T1": EU1, "EU_T2": EU2, "MI_T1": MI1, "MI_T2": MI2},
]


def load_or_build_priors(
    train_data: np.ndarray,
    train_seg: np.ndarray,
    feature_names: List[str],
    test_files: List[Path],
):
    if TRAIN_PRIORS_PATH.exists():
        print(f"[INFO] Loading train priors from {TRAIN_PRIORS_PATH} ...")
        data_npz = np.load(TRAIN_PRIORS_PATH, allow_pickle=True)

        var_order = data_npz["var_order"].tolist()
        min_vals = data_npz["min_vals"]
        max_vals = data_npz["max_vals"]
        diff_min_vals = data_npz["diff_min_vals"]
        diff_max_vals = data_npz["diff_max_vals"]
        var_min_vals = data_npz["var_min_vals"]
        var_max_vals = data_npz["var_max_vals"]
        feature_names = data_npz["feature_names"].tolist()
        rgb_db = list(data_npz["rgb_db"])

        eu_val_min = data_npz["eu_val_min"]
        eu_val_max = data_npz["eu_val_max"]
        eu_diff_min = data_npz["eu_diff_min"]
        eu_diff_max = data_npz["eu_diff_max"]
        eu_var_min = data_npz["eu_var_min"]
        eu_var_max = data_npz["eu_var_max"]

        mi_val_min = data_npz["mi_val_min"]
        mi_val_max = data_npz["mi_val_max"]
        mi_diff_min = data_npz["mi_diff_min"]
        mi_diff_max = data_npz["mi_diff_max"]
        mi_var_min = data_npz["mi_var_min"]
        mi_var_max = data_npz["mi_var_max"]

        print("[INFO] Train priors loaded from cache.")
    else:
        print("[INFO] No cache found. Building priors from train_data + global Min-Max ...")

        all_test_data = []
        all_test_seg = []
        all_test_windows_ext = []

        for f in test_files:
            d, _, seg = load_excel_with_boundary(f)
            all_test_data.append(d)
            all_test_seg.append(seg)

            win_ext_list, _ = sliding_windows_with_boundary(d, seg, WINDOW_SIZE, stride=WINDOW_STRIDE)
            all_test_windows_ext.extend(win_ext_list)

        if len(all_test_data) > 0:
            all_test_data = np.concatenate(all_test_data, axis=1)
            all_test_seg = np.concatenate(all_test_seg, axis=0)
        else:
            all_test_data = np.zeros_like(train_data)
            all_test_seg = np.zeros_like(train_seg)
            all_test_windows_ext = []

        concat_data = np.concatenate([train_data, all_test_data], axis=1)
        concat_seg = np.concatenate([train_seg, all_test_seg], axis=0)

        min_vals, max_vals = compute_minmax_stats(concat_data)
        diff_min_vals, diff_max_vals, var_min_vals, var_max_vals = compute_diff_var_minmax_stats(concat_data, concat_seg)

        (
            var_order,
            min_vals,
            max_vals,
            diff_min_vals,
            diff_max_vals,
            var_min_vals,
            var_max_vals,
            feature_names,
            rgb_db,
            eu_val_min, eu_val_max, eu_diff_min, eu_diff_max, eu_var_min, eu_var_max,
            mi_val_min, mi_val_max, mi_diff_min, mi_diff_max, mi_var_min, mi_var_max,
        ) = build_priors(
            train_data,
            train_seg,
            feature_names,
            min_vals,
            max_vals,
            diff_min_vals,
            diff_max_vals,
            var_min_vals,
            var_max_vals,
            all_test_windows_ext,
        )

        TRAIN_LIB_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            TRAIN_PRIORS_PATH,
            var_order=np.array(var_order, dtype=int),
            min_vals=min_vals,
            max_vals=max_vals,
            diff_min_vals=diff_min_vals,
            diff_max_vals=diff_max_vals,
            var_min_vals=var_min_vals,
            var_max_vals=var_max_vals,
            feature_names=np.array(feature_names),
            rgb_db=np.stack(rgb_db, axis=0),

            eu_val_min=eu_val_min, eu_val_max=eu_val_max,
            eu_diff_min=eu_diff_min, eu_diff_max=eu_diff_max,
            eu_var_min=eu_var_min, eu_var_max=eu_var_max,

            mi_val_min=mi_val_min, mi_val_max=mi_val_max,
            mi_diff_min=mi_diff_min, mi_diff_max=mi_diff_max,
            mi_var_min=mi_var_min, mi_var_max=mi_var_max,
        )
        print(f"[INFO] Train priors saved to {TRAIN_PRIORS_PATH}")

    return (
        var_order,
        feature_names,
        rgb_db,
        eu_val_min, eu_val_max, eu_diff_min, eu_diff_max, eu_var_min, eu_var_max,
        mi_val_min, mi_val_max, mi_diff_min, mi_diff_max, mi_var_min, mi_var_max,
        min_vals, max_vals, diff_min_vals, diff_max_vals, var_min_vals, var_max_vals,
    )


def save_additional_images_for_file(
    excel_path: Path,
    var_order: List[int],
    feature_names: List[str],
    rgb_db: List[np.ndarray],
    train_windows_ord: List[np.ndarray],
    eu_val_min: np.ndarray,
    eu_val_max: np.ndarray,
    eu_diff_min: np.ndarray,
    eu_diff_max: np.ndarray,
    eu_var_min: np.ndarray,
    eu_var_max: np.ndarray,
    mi_val_min: np.ndarray,
    mi_val_max: np.ndarray,
    mi_diff_min: np.ndarray,
    mi_diff_max: np.ndarray,
    mi_var_min: np.ndarray,
    mi_var_max: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    diff_min_vals: np.ndarray,
    diff_max_vals: np.ndarray,
    var_min_vals: np.ndarray,
    var_max_vals: np.ndarray,
    EU_T1: float,
    EU_T2: float,
    MI_T1: float,
    MI_T2: float,
):
    EU_SOFT_DIR_NAME = f"soft_threshold_EU_{EU_T1:.2f}_{EU_T2:.2f}_images"
    MI_SOFT_DIR_NAME = f"soft_threshold_MI_{MI_T1:.2f}_{MI_T2:.2f}_images"

    print(f"[SAVE] Processing extra images for file: {excel_path.name}")

    data, labels, seg = load_excel_with_boundary(excel_path)
    windows_ext, idx_pairs = sliding_windows_with_boundary(data, seg, WINDOW_SIZE, WINDOW_STRIDE)

    out_dir = OUTPUT_DIR / excel_path.stem
    eu_st_dir = out_dir / EU_SOFT_DIR_NAME
    mi_st_dir = out_dir / MI_SOFT_DIR_NAME

    eu_st_dir.mkdir(parents=True, exist_ok=True)
    mi_st_dir.mkdir(parents=True, exist_ok=True)

    ordered_names = [feature_names[i] for i in var_order]

    # range for EU normalize
    eu_val_range = eu_val_max - eu_val_min
    eu_diff_range = eu_diff_max - eu_diff_min
    eu_var_range = eu_var_max - eu_var_min
    eps = 1e-9
    eu_val_range[eu_val_range < eps] = 1.0
    eu_diff_range[eu_diff_range < eps] = 1.0
    eu_var_range[eu_var_range < eps] = 1.0

    # range for MI normalize
    mi_val_range = mi_val_max - mi_val_min
    mi_diff_range = mi_diff_max - mi_diff_min
    mi_var_range = mi_var_max - mi_var_min
    mi_val_range[mi_val_range < eps] = 1.0
    mi_diff_range[mi_diff_range < eps] = 1.0
    mi_var_range[mi_var_range < eps] = 1.0

    window_infos: List[Dict[str, Any]] = []

    for wid, (w_ext, (st_idx, ed_idx)) in enumerate(zip(windows_ext, idx_pairs)):
        print(f"[SAVE] File {excel_path.name}, window {wid} [{st_idx}, {ed_idx}]")

        w_full_ord = w_ext[var_order, :]
        if w_full_ord.shape[1] > WINDOW_SIZE:
            w_full_ord = w_full_ord[:, :WINDOW_SIZE]
        D_ord, W = w_full_ord.shape

        rgb = window_to_rgb(
            w_ext,
            var_order,
            min_vals,
            max_vals,
            diff_min_vals,
            diff_max_vals,
            var_min_vals,
            var_max_vals,
        ).astype(np.float32)

        best_idx, best_dist = find_most_similar_rgb_window(rgb_db, rgb)
        print(f"[SAVE] Best normal RGB window index = {best_idx}, RGB MSE distance = {best_dist:.4f}")

        Mv_diff_eu = np.zeros((D_ord, D_ord), dtype=float)
        Md_diff_eu = np.zeros((D_ord, D_ord), dtype=float)
        Mvar_diff_eu = np.zeros((D_ord, D_ord), dtype=float)
        Mv_diff_mi = np.zeros((D_ord, D_ord), dtype=float)
        Md_diff_mi = np.zeros((D_ord, D_ord), dtype=float)
        Mvar_diff_mi = np.zeros((D_ord, D_ord), dtype=float)

        ts_diff_val = np.zeros((D_ord, W), dtype=float)
        ts_diff_diff = np.zeros((D_ord, W), dtype=float)
        ts_diff_var = np.zeros((D_ord, W), dtype=float)

        if 0 <= best_idx < len(train_windows_ord):
            w_train_ord = train_windows_ord[best_idx]
            if w_train_ord.shape[1] > WINDOW_SIZE:
                w_train_ord = w_train_ord[:, :WINDOW_SIZE]

            # EU
            Mv_n_eu, Md_n_eu, Mvar_n_eu = compute_mean_rel_matrices(w_train_ord)
            Mv_t_eu, Md_t_eu, Mvar_t_eu = compute_mean_rel_matrices(w_full_ord)

            Mv_n_n = np.clip((Mv_n_eu - eu_val_min) / eu_val_range, 0.0, 1.0)
            Mv_t_n = np.clip((Mv_t_eu - eu_val_min) / eu_val_range, 0.0, 1.0)
            Md_n_n = np.clip((Md_n_eu - eu_diff_min) / eu_diff_range, 0.0, 1.0)
            Md_t_n = np.clip((Md_t_eu - eu_diff_min) / eu_diff_range, 0.0, 1.0)
            Mvar_n_n = np.clip((Mvar_n_eu - eu_var_min) / eu_var_range, 0.0, 1.0)
            Mvar_t_n = np.clip((Mvar_t_eu - eu_var_min) / eu_var_range, 0.0, 1.0)

            Mv_diff_eu = Mv_t_n - Mv_n_n
            Md_diff_eu = Md_t_n - Md_n_n
            Mvar_diff_eu = Mvar_t_n - Mvar_n_n

            # MI
            Mv_n_mi, Md_n_mi, Mvar_n_mi = compute_mi_matrices(w_train_ord)
            Mv_t_mi, Md_t_mi, Mvar_t_mi = compute_mi_matrices(w_full_ord)

            Mv_n_mi_n = np.clip((Mv_n_mi - mi_val_min) / mi_val_range, 0.0, 1.0)
            Mv_t_mi_n = np.clip((Mv_t_mi - mi_val_min) / mi_val_range, 0.0, 1.0)
            Md_n_mi_n = np.clip((Md_n_mi - mi_diff_min) / mi_diff_range, 0.0, 1.0)
            Md_t_mi_n = np.clip((Md_t_mi - mi_diff_min) / mi_diff_range, 0.0, 1.0)
            Mvar_n_mi_n = np.clip((Mvar_n_mi - mi_var_min) / mi_var_range, 0.0, 1.0)
            Mvar_t_mi_n = np.clip((Mvar_t_mi - mi_var_min) / mi_var_range, 0.0, 1.0)

            Mv_diff_mi = Mv_t_mi_n - Mv_n_mi_n
            Md_diff_mi = Md_t_mi_n - Md_n_mi_n
            Mvar_diff_mi = Mvar_t_mi_n - Mvar_n_mi_n


        Mv_diff_eu_st = soft_threshold_diff(Mv_diff_eu, t1=EU_T1, t2=EU_T2)
        Md_diff_eu_st = soft_threshold_diff(Md_diff_eu, t1=EU_T1, t2=EU_T2)
        Mvar_diff_eu_st = soft_threshold_diff(Mvar_diff_eu, t1=EU_T1, t2=EU_T2)

        Mv_diff_mi_st = soft_threshold_diff(Mv_diff_mi, t1=MI_T1, t2=MI_T2)
        Md_diff_mi_st = soft_threshold_diff(Md_diff_mi, t1=MI_T1, t2=MI_T2)
        Mvar_diff_mi_st = soft_threshold_diff(Mvar_diff_mi, t1=MI_T1, t2=MI_T2)

        save_corr_heatmap(
            Mv_diff_eu_st,
            eu_st_dir / f"w_{wid:05d}_eu_diff_value_test.png",
            var_names=ordered_names,
            title=f"[Test] Euclidean diff (value, soft, w={wid})",
        )
        save_corr_heatmap(
            Md_diff_eu_st,
            eu_st_dir / f"w_{wid:05d}_eu_diff_diff_test.png",
            var_names=ordered_names,
            title=f"[Test] Euclidean diff (diff, soft, w={wid})",
        )
        save_corr_heatmap(
            Mvar_diff_eu_st,
            eu_st_dir / f"w_{wid:05d}_eu_diff_var_test.png",
            var_names=ordered_names,
            title=f"[Test] Euclidean diff (var, soft, w={wid})",
        )

        save_corr_heatmap(
            Mv_diff_mi_st,
            mi_st_dir / f"w_{wid:05d}_mi_diff_value_test.png",
            var_names=ordered_names,
            title=f"[Test] MI diff (value, soft, w={wid})",
        )
        save_corr_heatmap(
            Md_diff_mi_st,
            mi_st_dir / f"w_{wid:05d}_mi_diff_diff_test.png",
            var_names=ordered_names,
            title=f"[Test] MI diff (diff, soft, w={wid})",
        )
        save_corr_heatmap(
            Mvar_diff_mi_st,
            mi_st_dir / f"w_{wid:05d}_mi_diff_var_test.png",
            var_names=ordered_names,
            title=f"[Test] MI diff (var, soft, w={wid})",
        )

        window_infos.append(
            {
                "wid": int(wid),
                "start": int(st_idx),
                "end": int(ed_idx),
                "best_idx": int(best_idx),
                "base_name_norm": f"train_w_{best_idx:05d}",
            }
        )

    meta_path = out_dir / "window_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"windows": window_infos}, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Done for file: {excel_path.name} | meta saved -> {meta_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    test_files = [p for p in DATA_DIR.glob("*.xlsx") if p.name != "train_data.xlsx"]

    train_data, _, train_seg = load_excel_with_boundary(TRAIN_PATH)
    df_train = pd.read_excel(TRAIN_PATH)
    feature_names = df_train.columns[:NUM_VARS].tolist()

    (
        var_order,
        feature_names,
        rgb_db,
        eu_val_min, eu_val_max, eu_diff_min, eu_diff_max, eu_var_min, eu_var_max,
        mi_val_min, mi_val_max, mi_diff_min, mi_diff_max, mi_var_min, mi_var_max,
        min_vals, max_vals, diff_min_vals, diff_max_vals, var_min_vals, var_max_vals,
    ) = load_or_build_priors(train_data, train_seg, feature_names, test_files)

    train_windows_ext, _ = sliding_windows_with_boundary(train_data, train_seg, WINDOW_SIZE, stride=WINDOW_STRIDE)
    train_windows_ord = [w_ext[var_order, :] for w_ext in train_windows_ext]

    for i, config in enumerate(PARAM_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Running parameter config {i+1}/{len(PARAM_CONFIGS)}: {config}")
        print(f"{'='*60}\n")

        EU_T1 = config["EU_T1"]
        EU_T2 = config["EU_T2"]
        MI_T1 = config["MI_T1"]
        MI_T2 = config["MI_T2"]

        for f in sorted(test_files):
            save_additional_images_for_file(
                f,
                var_order,
                feature_names,
                rgb_db,
                train_windows_ord,
                eu_val_min, eu_val_max, eu_diff_min, eu_diff_max, eu_var_min, eu_var_max,
                mi_val_min, mi_val_max, mi_diff_min, mi_diff_max, mi_var_min, mi_var_max,
                min_vals, max_vals, diff_min_vals, diff_max_vals, var_min_vals, var_max_vals,
                EU_T1=EU_T1, EU_T2=EU_T2,
                MI_T1=MI_T1, MI_T2=MI_T2,
            )


if __name__ == "__main__":
    main()
