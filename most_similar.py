# most_similar.py
from pathlib import Path
from typing import List, Tuple

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
    compute_minmax_stats,
    compute_diff_var_minmax_stats,
)
from main import build_priors  # 只用它来构建训练库

TRAIN_IMG_DIR = TRAIN_LIB_DIR / "images"
TRAIN_PRIORS_PATH = TRAIN_LIB_DIR / "train_priors.npz"


def dump_similarity_for_dataset(
    excel_path: Path,
    var_order: List[int],
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    diff_min_vals: np.ndarray,
    diff_max_vals: np.ndarray,
    var_min_vals: np.ndarray,
    var_max_vals: np.ndarray,
    rgb_db: List[np.ndarray],
):
    """
    对单个测试集（一个 excel 文件）：
      - 仅在 OUTPUT_DIR/<stem>/ 目录存在时才处理
      - 如果 OUTPUT_DIR/<stem>/rgb_similarity.xlsx 已存在，则跳过
      - 否则：
          * 对每个测试窗口生成 RGB
          * 与所有训练窗口计算 MSE
          * 以矩阵形式保存到 rgb_similarity.xlsx：
                行：训练窗口
                列：测试窗口
                单元格：MSE
          * 第二个 sheet 记录每个测试窗口最相似的训练窗口
    """
    out_dir = OUTPUT_DIR / excel_path.stem
    if not out_dir.exists():
        print(f"[WARN] Skip {excel_path.name}: output folder {out_dir} not found.")
        return

    sim_xlsx = out_dir / "rgb_similarity.xlsx"
    if sim_xlsx.exists():
        print(f"[INFO] Skip {excel_path.name}: {sim_xlsx} already exists.")
        return

    print(f"[INFO] Processing similarity for dataset: {excel_path.name}")

    # 读取测试数据
    data, labels, seg = load_excel_with_boundary(excel_path)
    # 测试集滑窗：步长使用 WINDOW_STRIDE（和主推理一致）
    windows_ext, idx_pairs = sliding_windows_with_boundary(
        data, seg, WINDOW_SIZE, WINDOW_STRIDE
    )

    if len(windows_ext) == 0:
        print(f"[WARN] No windows found for {excel_path.name}, skip.")
        return

    n_train = len(rgb_db)
    n_test = len(windows_ext)

    # 初始化 MSE 矩阵：行 = 训练窗口，列 = 测试窗口
    mse_mat = np.full((n_train, n_test), np.nan, dtype=float)

    for wid, w_ext in enumerate(windows_ext):
        # 生成测试窗口的 RGB（与主流程完全一致）
        rgb_test = window_to_rgb(
            w_ext,
            var_order,
            min_vals,
            max_vals,
            diff_min_vals,
            diff_max_vals,
            var_min_vals,
            var_max_vals,
        )
        rgb_test_f = rgb_test.astype(float)

        # 与所有训练窗口计算 MSE
        for train_idx, rgb_train in enumerate(rgb_db):
            if rgb_train.shape != rgb_test.shape:
                # 理论上不应该出现形状不一致，如果出现就直接跳过这个训练窗口
                continue
            diff = rgb_train.astype(float) - rgb_test_f
            mse = np.mean(diff * diff)
            mse_mat[train_idx, wid] = mse

    # ---- 构建矩阵 DataFrame ----
    row_index = [f"train_{i:05d}" for i in range(n_train)]
    col_names = [f"test_{j:05d}" for j in range(n_test)]
    df_mat = pd.DataFrame(mse_mat, index=row_index, columns=col_names)

    # ---- 为每个测试窗口找最相似训练窗口 ----
    best_records = []
    for j in range(n_test):
        col = df_mat.iloc[:, j]
        # 跳过全 NaN 的列
        if not col.notna().any():
            continue
        best_row_label = col.idxmin()          # 例如 "train_00495"
        best_mse = col.loc[best_row_label]
        try:
            best_train_id = int(best_row_label.split("_")[1])
        except Exception:
            # 万一命名改了，就退回到行号
            best_train_id = df_mat.index.get_loc(best_row_label)

        best_records.append(
            {
                "test_window_id": j,
                "best_train_window_id": best_train_id,
                "best_train_row_label": best_row_label,
                "best_mse": best_mse,
            }
        )

    best_df = pd.DataFrame(best_records)

    # ---- 写入 Excel：两个 sheet ----
    with pd.ExcelWriter(sim_xlsx) as writer:
        df_mat.to_excel(writer, sheet_name="mse_matrix", index=True)
        best_df.to_excel(writer, sheet_name="best_match", index=False)

    print(f"[OK] Saved similarity excel → {sim_xlsx}")


def load_or_build_priors_for_similarity():
    """
    和 main.main 里的逻辑保持一致：
      - 若 train_priors.npz 存在，则直接加载
      - 否则：
          * 读训练集 + 所有测试集
          * 计算“训练+测试”的全局 Min-Max
          * 调用 build_priors(训练集, 全局MinMax) 构建训练库
          * 保存 npz 缓存
    返回：
      (var_order, min_vals, max_vals,
       diff_min_vals, diff_max_vals,
       var_min_vals, var_max_vals,
       rgb_db, test_files)
    """
    # 列出测试文件
    test_files = [p for p in DATA_DIR.glob("*.xlsx") if p.name != "train_data.xlsx"]

    if TRAIN_PRIORS_PATH.exists():
        print(f"[INFO] Loading train priors from {TRAIN_PRIORS_PATH} ...")
        data_npz = np.load(TRAIN_PRIORS_PATH, allow_pickle=True)

        var_order     = data_npz["var_order"].tolist()
        min_vals      = data_npz["min_vals"]
        max_vals      = data_npz["max_vals"]
        diff_min_vals = data_npz["diff_min_vals"]
        diff_max_vals = data_npz["diff_max_vals"]
        var_min_vals  = data_npz["var_min_vals"]
        var_max_vals  = data_npz["var_max_vals"]
        rgb_db        = list(data_npz["rgb_db"])

        print("[INFO] Train priors loaded from cache.")
        return (
            var_order,
            min_vals,
            max_vals,
            diff_min_vals,
            diff_max_vals,
            var_min_vals,
            var_max_vals,
            rgb_db,
            test_files,
        )

    # ===== 没有缓存，需要现算（逻辑和 main.py 中一致） =====
    print("[INFO] No cache found. Building priors from train_data + all test files with global Min-Max ...")

    # 1) 读取训练集
    train_data, train_labels, train_seg = load_excel_with_boundary(TRAIN_PATH)
    df_train = pd.read_excel(TRAIN_PATH)
    feature_names = df_train.columns[:NUM_VARS].tolist()

    # 2) 收集全部测试集数据，用于全局 Min-Max
    all_test_data = []
    all_test_seg = []

    for f in test_files:
        d, lbl, seg = load_excel_with_boundary(f)
        all_test_data.append(d)
        all_test_seg.append(seg)

    if len(all_test_data) > 0:
        all_test_data = np.concatenate(all_test_data, axis=1)
        all_test_seg = np.concatenate(all_test_seg, axis=0)
    else:
        all_test_data = np.zeros_like(train_data)
        all_test_seg = np.zeros_like(train_labels)

    # 3) 计算“训练 + 测试”的全局 Min-Max
    concat_data = np.concatenate([train_data, all_test_data], axis=1)
    concat_seg = np.concatenate([train_seg, all_test_seg], axis=0)

    min_vals, max_vals = compute_minmax_stats(concat_data)
    diff_min_vals, diff_max_vals, var_min_vals, var_max_vals = \
        compute_diff_var_minmax_stats(concat_data, concat_seg)

    # 4) 用全局 Min-Max + 训练数据构建先验（调用 main.build_priors）
    (
        var_order,
        min_vals,
        max_vals,
        diff_min_vals,
        diff_max_vals,
        var_min_vals,
        var_max_vals,
        corr_val_db,
        corr_diff_db,
        corr_var_db,
        feature_names,
        rgb_db,
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
    )

    # 5) 保存先验到 npz，供下次直接加载
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
        corr_val_list_ord=np.stack(corr_val_db, axis=0),
        corr_diff_list_ord=np.stack(corr_diff_db, axis=0),
        corr_var_list_ord=np.stack(corr_var_db, axis=0),
        feature_names=np.array(feature_names),
        rgb_db=np.stack(rgb_db, axis=0),
    )
    print(f"[INFO] Train priors saved to {TRAIN_PRIORS_PATH}")

    return (
        var_order,
        min_vals,
        max_vals,
        diff_min_vals,
        diff_max_vals,
        var_min_vals,
        var_max_vals,
        rgb_db,
        test_files,
    )


def main():
    (
        var_order,
        min_vals,
        max_vals,
        diff_min_vals,
        diff_max_vals,
        var_min_vals,
        var_max_vals,
        rgb_db,
        test_files,
    ) = load_or_build_priors_for_similarity()

    if not test_files:
        print("[WARN] No test excel files found.")
        return

    for f in sorted(test_files):
        dump_similarity_for_dataset(
            f,
            var_order,
            min_vals,
            max_vals,
            diff_min_vals,
            diff_max_vals,
            var_min_vals,
            var_max_vals,
            rgb_db,
        )


if __name__ == "__main__":
    main()
