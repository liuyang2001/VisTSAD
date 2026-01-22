# compute_diff_max.py
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    TRAIN_PATH,
    OUTPUT_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    NUM_VARS,
    TRAIN_LIB_DIR,
)
from data_utils import (
    load_excel_with_boundary,
    sliding_windows_with_boundary,
    window_to_rgb,
    compute_mean_rel_matrices,
    compute_mi_matrices,
    find_most_similar_rgb_window,
)

TRAIN_PRIORS_PATH = TRAIN_LIB_DIR / "train_priors.npz"


def load_priors():
    """
    从 train_priors.npz 加载先验（与你 main.py 保存的一致）
    """
    if not TRAIN_PRIORS_PATH.exists():
        raise FileNotFoundError(
            f"train priors not found: {TRAIN_PRIORS_PATH}. "
            f"请先运行 main.py 生成 train_priors.npz。"
        )

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
        eu_val_min,
        eu_val_max,
        eu_diff_min,
        eu_diff_max,
        eu_var_min,
        eu_var_max,
        mi_val_min,
        mi_val_max,
        mi_diff_min,
        mi_diff_max,
        mi_var_min,
        mi_var_max,
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1) 加载先验
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
        eu_val_min,
        eu_val_max,
        eu_diff_min,
        eu_diff_max,
        eu_var_min,
        eu_var_max,
        mi_val_min,
        mi_val_max,
        mi_diff_min,
        mi_diff_max,
        mi_var_min,
        mi_var_max,
    ) = load_priors()

    # 2) 构建训练窗口（用于“最相似正常窗口”）
    train_data, train_labels, train_seg = load_excel_with_boundary(TRAIN_PATH)
    train_windows_ext, _ = sliding_windows_with_boundary(
        train_data, train_seg, WINDOW_SIZE, stride=WINDOW_STRIDE
    )
    train_windows_ord = [w_ext[var_order, :] for w_ext in train_windows_ext]

    # 3) 预备欧氏 / MI 的归一化 range
    eps = 1e-9
    eu_val_range = eu_val_max - eu_val_min
    eu_diff_range = eu_diff_max - eu_diff_min
    eu_var_range = eu_var_max - eu_var_min
    eu_val_range[eu_val_range < eps] = 1.0
    eu_diff_range[eu_diff_range < eps] = 1.0
    eu_var_range[eu_var_range < eps] = 1.0

    mi_val_range = mi_val_max - mi_val_min
    mi_diff_range = mi_diff_max - mi_diff_min
    mi_var_range = mi_var_max - mi_var_min
    mi_val_range[mi_val_range < eps] = 1.0
    mi_diff_range[mi_diff_range < eps] = 1.0
    mi_var_range[mi_var_range < eps] = 1.0

    # 4) 列出测试文件
    test_files = [p for p in DATA_DIR.glob("*.xlsx") if p.name != "train_data.xlsx"]

    all_rows: List[Dict[str, Any]] = []

    # 5) 遍历所有测试文件 & 窗口
    for f in sorted(test_files):
        print(f"[INFO] Processing test file for diff max: {f.name}")
        data, labels, seg = load_excel_with_boundary(f)
        windows_ext, idx_pairs = sliding_windows_with_boundary(
            data, seg, WINDOW_SIZE, WINDOW_STRIDE
        )

        for wid, (w_ext, (st, ed)) in enumerate(zip(windows_ext, idx_pairs)):
            print(f"[INFO]   window {wid} [{st}, {ed}]")

            # --- RGB，用于找最相似正常窗口 ---
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
            best_idx, best_dist = find_most_similar_rgb_window(rgb_db, rgb)

            # --- 计算欧氏 / MI 差值矩阵 ---
            w_full_ord = w_ext[var_order, :]
            D_ord, _ = w_full_ord.shape

            if best_idx < 0 or best_idx >= len(train_windows_ord):
                print(
                    f"[WARN] best_idx={best_idx} out of range, "
                    f"use zeros diff for this window."
                )
                Mv_diff_eu = np.zeros((D_ord, D_ord), dtype=float)
                Md_diff_eu = np.zeros((D_ord, D_ord), dtype=float)
                Mvar_diff_eu = np.zeros((D_ord, D_ord), dtype=float)
                Mv_diff_mi = np.zeros((D_ord, D_ord), dtype=float)
                Md_diff_mi = np.zeros((D_ord, D_ord), dtype=float)
                Mvar_diff_mi = np.zeros((D_ord, D_ord), dtype=float)
            else:
                w_train_ord = train_windows_ord[best_idx]

                # --- Euclidean matrices ---
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

                # --- MI matrices ---
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

            # --- 对每个窗口取最大欧氏差 & 最大 MI 差 ---
            eu_stack = np.stack(
                [np.abs(Mv_diff_eu), np.abs(Md_diff_eu), np.abs(Mvar_diff_eu)],
                axis=0,
            )  # (3, D, D)
            mi_stack = np.stack(
                [np.abs(Mv_diff_mi), np.abs(Md_diff_mi), np.abs(Mvar_diff_mi)],
                axis=0,
            )
            eu_max = float(np.max(eu_stack))
            mi_max = float(np.max(mi_stack))

            # --- 根据 wid 规则生成 Label(0/1) ---
            # 每个测试集内部：
            #   前 31 个窗口 wid 0-30 → Label = 0
            #   接着 31 个窗口 wid 31-61 → Label = 1
            #   再接着 31 个窗口 wid 62-92 → Label = 0
            #   再接着 31 个窗口 wid 93-123 → Label = 1
            #   …… 以此类推，每 31 个窗口一组，0/1 交替
            block_idx = wid // 31
            label_int = block_idx % 2  # 0 or 1

            all_rows.append(
                {
                    "file": f.name,
                    "file_stem": f.stem,
                    "wid": wid,
                    "start": st,
                    "end": ed,
                    "Label": int(label_int),  # 0 or 1
                    "eu_max": eu_max,
                    "mi_max": mi_max,
                }
            )

    # 6) 生成 DataFrame & 统计 Label=0/1 的最大值
    if not all_rows:
        print("[WARN] No windows processed. Nothing to save.")
        return

    df = pd.DataFrame(all_rows)

    # 按 Label 分组，取 eu_max / mi_max 的最大值
    summary_list = []
    for label_value in sorted(df["Label"].unique()):
        sub = df[df["Label"] == label_value]
        if len(sub) == 0:
            continue
        summary_list.append(
            {
                "Label": int(label_value),
                "eu_max_max": float(sub["eu_max"].max()),
                "mi_max_max": float(sub["mi_max"].max()),
                "num_windows": int(len(sub)),
            }
        )

    summary_df = pd.DataFrame(summary_list)

    # 7) 写入一个 Excel，两个 sheet
    out_path = OUTPUT_DIR / "diff_max_summary.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        df.to_excel(writer, sheet_name="window_max", index=False)
        summary_df.to_excel(writer, sheet_name="label_max", index=False)

    print(f"[OK] Saved window-level & label-level max diff → {out_path}")


if __name__ == "__main__":
    main()
