# main.py
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    compute_minmax_stats,
    compute_diff_var_minmax_stats,
    window_to_rgb,
    find_most_similar_rgb_window,
    compute_mean_rel_matrices,
    compute_mi_matrices,
    mi_based_order,
)

from vlm_utils import (
    build_direct_prompt_gray_plus_rel,
    # build_direct_prompt_rel_only,
    build_direct_prompt_gray_only,
    build_direct_prompt_gray_plus_eu,
    build_direct_prompt_gray_plus_mi,
    build_direct_prompt_eu_only,
    build_direct_prompt_mi_only,
    build_direct_prompt_eu_plus_mi,
    build_direct_prompt_value_only,
    call_vlm_gray_plus_rel,
    # call_vlm_rel_only,
    call_vlm_gray_only,
    call_vlm_gray_plus_eu,
    call_vlm_gray_plus_mi,
    call_vlm_eu_only,
    call_vlm_mi_only,
    call_vlm_eu_plus_mi,
    call_vlm_value_only
)

TRAIN_IMG_DIR = TRAIN_LIB_DIR / "images"
TRAIN_PRIORS_PATH = TRAIN_LIB_DIR / "train_priors.npz"


def fmt2(x: float) -> str:
    return f"{x:.2f}"


# ============================================================
# 关系图加载（EU/MI）
# ============================================================
def get_relation_image_paths(out_dir: Path, wid: int, eu_folder: str, mi_folder: str) -> Dict[str, Path]:
    eu_dir = out_dir / eu_folder
    mi_dir = out_dir / mi_folder
    # print(f"relation images:")
    # print(eu_dir / f"w_{wid:05d}_eu_diff_value_test.png")
    # print(eu_dir / f"w_{wid:05d}_eu_diff_diff_test.png")
    # print(eu_dir / f"w_{wid:05d}_eu_diff_var_test.png")
    # print(mi_dir / f"w_{wid:05d}_mi_diff_value_test.png")
    # print(mi_dir / f"w_{wid:05d}_mi_diff_diff_test.png")
    # print(mi_dir / f"w_{wid:05d}_mi_diff_var_test.png")
    return {
        "eu_val": eu_dir / f"w_{wid:05d}_eu_diff_value_test.png",
        "eu_diff": eu_dir / f"w_{wid:05d}_eu_diff_diff_test.png",
        "eu_var": eu_dir / f"w_{wid:05d}_eu_diff_var_test.png",
        "mi_val": mi_dir / f"w_{wid:05d}_mi_diff_value_test.png",
        "mi_diff": mi_dir / f"w_{wid:05d}_mi_diff_diff_test.png",
        "mi_var": mi_dir / f"w_{wid:05d}_mi_diff_var_test.png",
    }
def get_eu_relation_image_paths(out_dir: Path, wid: int, eu_folder: str, mi_folder: str) -> Dict[str, Path]:
    eu_dir = out_dir / eu_folder
    # print(f"relation images:")
    # print(eu_dir / f"w_{wid:05d}_eu_diff_value_test.png")
    # print(eu_dir / f"w_{wid:05d}_eu_diff_diff_test.png")
    # print(eu_dir / f"w_{wid:05d}_eu_diff_var_test.png")
    return {
        "eu_val": eu_dir / f"w_{wid:05d}_eu_diff_value_test.png",
        "eu_diff": eu_dir / f"w_{wid:05d}_eu_diff_diff_test.png",
        "eu_var": eu_dir / f"w_{wid:05d}_eu_diff_var_test.png",
    }
def get_mi_relation_image_paths(out_dir: Path, wid: int, eu_folder: str, mi_folder: str) -> Dict[str, Path]:
    eu_dir = out_dir / eu_folder
    mi_dir = out_dir / mi_folder
    # print(f"relation images:")
    # print(mi_dir / f"w_{wid:05d}_mi_diff_value_test.png")
    # print(mi_dir / f"w_{wid:05d}_mi_diff_diff_test.png")
    # print(mi_dir / f"w_{wid:05d}_mi_diff_var_test.png")
    return {
        "mi_val": mi_dir / f"w_{wid:05d}_mi_diff_value_test.png",
        "mi_diff": mi_dir / f"w_{wid:05d}_mi_diff_diff_test.png",
        "mi_var": mi_dir / f"w_{wid:05d}_mi_diff_var_test.png",
    }


# ============================================================
# 时序差值图（GRAY soft-threshold）加载
# ============================================================
def get_ts_diff_image_paths(out_dir: Path, wid: int, gray_folder: str) -> Dict[str, Path]:
    gdir = out_dir / gray_folder
    # print(f"gray images:")
    # print(gdir / f"w_{wid:05d}_ts_diff_value_test.png")
    # print(gdir / f"w_{wid:05d}_ts_diff_diff_test.png")
    # print(gdir / f"w_{wid:05d}_ts_diff_var_test.png")
    return {
        "ts_val": gdir / f"w_{wid:05d}_ts_diff_value_test.png",
        "ts_diff": gdir / f"w_{wid:05d}_ts_diff_diff_test.png",
        "ts_var": gdir / f"w_{wid:05d}_ts_diff_var_test.png",
    }


# ============================================================
# 训练先验（原样保留：RGB 库 + EU/MI 全局 min/max）
# ============================================================
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

    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
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
        # rgb_db.append(rgb.astype(np.uint8))
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


# ============================================================
# process（单个阈值组合 + 单个 mode）
# ============================================================
def process_file_one_setting(
    excel_path: Path,
    eu_folder: str,
    mi_folder: str,
    gray_folder: str,
    EU1: float,
    EU2: float,
    MI1: float,
    MI2: float,
    GRAY1: float,
    GRAY2: float,
    mode: str = "gray_plus_rel",
):
    # assert mode in ("gray_plus_rel", "rel_only", "gray_only")
    assert mode in ("gray_plus_rel", "gray_only", "gray_plus_eu", "gray_plus_mi","eu_only","mi_only","eu_plus_mi","value_only")
    EU1s, EU2s, MI1s, MI2s = fmt2(EU1), fmt2(EU2), fmt2(MI1), fmt2(MI2)
    G1s, G2s = fmt2(GRAY1), fmt2(GRAY2)

    tag = f"EU={EU1s}-{EU2s}__MI={MI1s}-{MI2s}__G={G1s}-{G2s}"
    print(f"[INFO] Processing file: {excel_path.name} | {tag} | mode={mode}")

    data, labels, seg = load_excel_with_boundary(excel_path)
    out_dir = OUTPUT_DIR / excel_path.stem

    # if mode == "rel_only":
    #     json_dir = out_dir / f"vlm_rel_only_json__{tag}"
    # elif mode == "gray_only":
    #     json_dir = out_dir / f"vlm_gray_only_json__{tag}"
    # else:
    #     json_dir = out_dir / f"vlm_gray_json__{tag}"
    if mode == "gray_only":
        json_dir = out_dir / f"vlm_gray_only_json__{tag}"
    elif mode == "gray_plus_eu":
        json_dir = out_dir / f"vlm_gray_plus_eu_json__{tag}"
    elif mode == "gray_plus_mi":
        json_dir = out_dir / f"vlm_gray_plus_mi_json__{tag}"
    elif mode == "eu_only":
        json_dir = out_dir / f"vlm_eu_only_json__{tag}"
    elif mode == "mi_only":
        json_dir = out_dir / f"vlm_mi_only_json__{tag}"
    elif mode == "eu_plus_mi":
        json_dir = out_dir / f"vlm_eu_plus_mi_json__{tag}"
    elif mode == "value_only":
        json_dir = out_dir / f"vlm_value_only_json__{tag}"
    else:  # gray_plus_rel
        json_dir = out_dir / f"vlm_gray_json__{tag}"
    json_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "window_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[ERROR] window_meta.json not found for {excel_path.name}. "
            f"Run save_extra_images.py first to generate images+meta."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    window_infos: List[Dict[str, Any]] = sorted(meta.get("windows", []), key=lambda x: x["wid"])

    # if mode == "rel_only":
    #     prompt = build_direct_prompt_rel_only()
    # elif mode == "gray_only":
    #     prompt = build_direct_prompt_gray_only()
    # else:
    #     prompt = build_direct_prompt_gray_plus_rel()
    if mode == "gray_only":
        prompt = build_direct_prompt_gray_only()          # 但会按 gray_plus_rel 风格改写（见后面 vlm_utils.py）
    elif mode == "gray_plus_eu":
        prompt = build_direct_prompt_gray_plus_eu()
    elif mode == "gray_plus_mi":
        prompt = build_direct_prompt_gray_plus_mi()
    elif mode == "eu_only":
        prompt = build_direct_prompt_eu_only()
    elif mode == "mi_only":
        prompt = build_direct_prompt_mi_only()
    elif mode == "eu_plus_mi":
        prompt = build_direct_prompt_eu_plus_mi()
    elif mode == "value_only":
        prompt = build_direct_prompt_value_only()
    else:
        prompt = build_direct_prompt_gray_plus_rel()


    win_records: List[Dict[str, Any]] = []
    to_process: List[Dict[str, Any]] = []

    # cache 复用
    for info in window_infos:
        wid = int(info["wid"])
        st = int(info["start"])
        ed = int(info["end"])
        json_path = json_dir / f"vlm_reply_w_{wid:05d}.json"

        if json_path.exists():
            print(f"[INFO] Reuse cached VLM json: {json_path.name}")
            try:
                obj = json.loads(json_path.read_text(encoding="utf-8"))

                raw_label = obj.get("Label", 0)
                if isinstance(raw_label, list):
                    tmp = []
                    for x in raw_label:
                        try:
                            v = int(x)
                        except Exception:
                            v = 0
                        tmp.append(1 if v == 1 else 0)
                    label_seq = tmp[:WINDOW_SIZE] if len(tmp) >= WINDOW_SIZE else tmp + [0] * (WINDOW_SIZE - len(tmp))
                else:
                    try:
                        v = int(raw_label)
                    except Exception:
                        v = 0
                    v = 1 if v == 1 else 0
                    label_seq = [v] * WINDOW_SIZE

                label_int = 1 if any(label_seq) else 0

                rc = obj.get("root_cause_variables", [])
                if isinstance(rc, str):
                    root_causes = [rc]
                elif isinstance(rc, list):
                    root_causes = [str(x) for x in rc]
                else:
                    root_causes = []

                raw = obj.get("Analysis Process", "")

                true_seq = labels[st:ed + 1]
                true_window_label = int((true_seq == 1).any())
                true_seq_str = "".join(str(int(x)) for x in true_seq)
                pred_seq_str = "".join(str(int(x)) for x in label_seq)

                win_records.append(
                    {
                        "file": excel_path.name,
                        "wid": wid,
                        "start": st,
                        "end": ed,
                        "EU1": EU1s, "EU2": EU2s, "MI1": MI1s, "MI2": MI2s, "GRAY1": G1s, "GRAY2": G2s,
                        "EU_folder": eu_folder,
                        "MI_folder": mi_folder,
                        "GRAY_folder": gray_folder,
                        "mode": mode,
                        "true_window_label": true_window_label,
                        "true_label_seq": true_seq_str,
                        "pred_label_seq": pred_seq_str,
                        "pred_is_anomaly": int(label_int),
                        "pred_label": "anomaly" if label_int else "normal",
                        "root_cause_variables": ",".join(root_causes) if root_causes else "",
                        "raw": raw,
                    }
                )
            except Exception as e:
                print(f"[WARN] Failed to reuse cache for wid={wid}: {e}")
                to_process.append(info)
        else:
            to_process.append(info)

    def _run_one_window(info: Dict[str, Any]) -> Dict[str, Any]:
        wid = int(info["wid"])
        st = int(info["start"])
        ed = int(info["end"])
        json_path = json_dir / f"vlm_reply_w_{wid:05d}.json"

        rel_paths = get_relation_image_paths(out_dir, wid, eu_folder, mi_folder)
        ts_paths = get_ts_diff_image_paths(out_dir, wid, gray_folder)
        eu_paths = get_eu_relation_image_paths(out_dir, wid, eu_folder, mi_folder)
        mi_paths = get_mi_relation_image_paths(out_dir, wid, eu_folder, mi_folder)

        # if mode in ("gray_plus_rel"):
        #     for k, p in rel_paths.items():
        #         if not p.exists():
        #             raise FileNotFoundError(f"[ERROR] Missing relation map: {p} (key={k})")
        if mode in ("gray_plus_rel","gray_plus_eu","eu_only","eu_plus_mi","value_only"):
            for k, p in eu_paths.items():
                if not p.exists():
                    raise FileNotFoundError(f"[ERROR] Missing eu relation map: {p} (key={k})")
        if mode in ("gray_plus_rel", "gray_plus_mi","mi_only","eu_plus_mi","value_only"):
            for k, p in mi_paths.items():
                if not p.exists():
                    raise FileNotFoundError(f"[ERROR] Missing mi relation map: {p} (key={k})")
        if mode in ("gray_plus_rel", "gray_only", "gray_plus_eu", "gray_plus_mi"):
            for k, p in ts_paths.items():
                if not p.exists():
                    raise FileNotFoundError(
                        f"[ERROR] Missing time-series diff map: {p} (key={k}). "
                        f"Run save_extra_images.py first."
                    )

        t0 = time.perf_counter()

        # if mode == "rel_only":
        #     label_int_tmp, root_causes_tmp, raw = call_vlm_rel_only(
        #         prompt=prompt,
        #         img_eu_diff_val_test=rel_paths["eu_val"],
        #         img_eu_diff_diff_test=rel_paths["eu_diff"],
        #         img_eu_diff_var_test=rel_paths["eu_var"],
        #         img_mi_diff_val_test=rel_paths["mi_val"],
        #         img_mi_diff_diff_test=rel_paths["mi_diff"],
        #         img_mi_diff_var_test=rel_paths["mi_var"],
        #         save_dir=json_dir,
        #         window_id=wid,
        #         normal_window_idx=int(info["best_idx"]),
        #         normal_window_name=str(info["base_name_norm"]),
        #         eu_folder=eu_folder,
        #         mi_folder=mi_folder,
        #     )

        if mode == "gray_only":
            label_int_tmp, root_causes_tmp, raw = call_vlm_gray_only(
                prompt=prompt,
                img_diff_value=ts_paths["ts_val"],
                img_diff_diff=ts_paths["ts_diff"],
                img_diff_var=ts_paths["ts_var"],
                save_dir=json_dir,
                window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder,
                mi_folder=mi_folder,
            )
        elif mode == "gray_plus_eu":
            label_int_tmp, root_causes_tmp, raw = call_vlm_gray_plus_eu(
                prompt=prompt,
                img_diff_value=ts_paths["ts_val"],
                img_diff_diff=ts_paths["ts_diff"],
                img_diff_var=ts_paths["ts_var"],
                img_eu_diff_val_test=rel_paths["eu_val"],
                img_eu_diff_diff_test=rel_paths["eu_diff"],
                img_eu_diff_var_test=rel_paths["eu_var"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,  # 你现在 json 里也存了，继续存没问题
            )
        elif mode == "gray_plus_mi":
            label_int_tmp, root_causes_tmp, raw = call_vlm_gray_plus_mi(
                prompt=prompt,
                img_diff_value=ts_paths["ts_val"],
                img_diff_diff=ts_paths["ts_diff"],
                img_diff_var=ts_paths["ts_var"],
                img_mi_diff_val_test=rel_paths["mi_val"],
                img_mi_diff_diff_test=rel_paths["mi_diff"],
                img_mi_diff_var_test=rel_paths["mi_var"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,
            )
        elif mode == "eu_only":
            label_int_tmp, root_causes_tmp, raw = call_vlm_eu_only(
                prompt=prompt,
                img_eu_diff_val_test=rel_paths["eu_val"],
                img_eu_diff_diff_test=rel_paths["eu_diff"],
                img_eu_diff_var_test=rel_paths["eu_var"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,  # 你现在 json 里也存了，继续存没问题
            )
        elif mode == "mi_only":
            label_int_tmp, root_causes_tmp, raw = call_vlm_mi_only(
                prompt=prompt,
                img_mi_diff_val_test=rel_paths["mi_val"],
                img_mi_diff_diff_test=rel_paths["mi_diff"],
                img_mi_diff_var_test=rel_paths["mi_var"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,  # 你现在 json 里也存了，继续存没问题
            )
        elif mode == "eu_plus_mi":
            label_int_tmp, root_causes_tmp, raw = call_vlm_eu_plus_mi(
                prompt=prompt,
                img_eu_diff_val_test=rel_paths["eu_val"],
                img_eu_diff_diff_test=rel_paths["eu_diff"],
                img_eu_diff_var_test=rel_paths["eu_var"],
                img_mi_diff_val_test=rel_paths["mi_val"],
                img_mi_diff_diff_test=rel_paths["mi_diff"],
                img_mi_diff_var_test=rel_paths["mi_var"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,  # 你现在 json 里也存了，继续存没问题
            )
        elif mode == "value_only":
            label_int_tmp, root_causes_tmp, raw = call_vlm_value_only(
                prompt=prompt,
                img_eu_diff_val_test=rel_paths["eu_val"],
                # img_eu_diff_diff_test=rel_paths["eu_diff"],
                # img_eu_diff_var_test=rel_paths["eu_var"],
                img_mi_diff_val_test=rel_paths["mi_val"],
                # img_mi_diff_diff_test=rel_paths["mi_diff"],
                # img_mi_diff_var_test=rel_paths["mi_var"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,  # 你现在 json 里也存了，继续存没问题
            )
        else:
            label_int_tmp, root_causes_tmp, raw = call_vlm_gray_plus_rel(
                prompt=prompt,
                img_diff_value=ts_paths["ts_val"],
                img_diff_diff=ts_paths["ts_diff"],
                img_diff_var=ts_paths["ts_var"],
                img_eu_diff_val_test=rel_paths["eu_val"],
                img_eu_diff_diff_test=rel_paths["eu_diff"],
                img_eu_diff_var_test=rel_paths["eu_var"],
                img_mi_diff_val_test=rel_paths["mi_val"],
                img_mi_diff_diff_test=rel_paths["mi_diff"],
                img_mi_diff_var_test=rel_paths["mi_var"],
                save_dir=json_dir,
                window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder,
                mi_folder=mi_folder,
            )

        elapsed = time.perf_counter() - t0

        # readback（保持你原来的兼容逻辑）
        label_seq = [0] * WINDOW_SIZE
        root_causes = root_causes_tmp
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
            raw_label = obj.get("Label", 0)
            if isinstance(raw_label, list):
                tmp = []
                for x in raw_label:
                    try:
                        v = int(x)
                    except Exception:
                        v = 0
                    tmp.append(1 if v == 1 else 0)
                label_seq = tmp[:WINDOW_SIZE] if len(tmp) >= WINDOW_SIZE else tmp + [0] * (WINDOW_SIZE - len(tmp))
            else:
                try:
                    v = int(raw_label)
                except Exception:
                    v = 0
                v = 1 if v == 1 else 0
                label_seq = [v] * WINDOW_SIZE

            rc = obj.get("root_cause_variables", [])
            if isinstance(rc, str):
                root_causes = [rc]
            elif isinstance(rc, list):
                root_causes = [str(x) for x in rc]
            else:
                root_causes = []
        except Exception as e:
            print(f"[WARN] Failed to read back JSON for wid={wid}: {e}")
            label_seq = [label_int_tmp] * WINDOW_SIZE

        label_int = 1 if any(label_seq) else 0

        true_seq = labels[st:ed + 1]
        true_window_label = int((true_seq == 1).any())
        true_seq_str = "".join(str(int(x)) for x in true_seq)
        pred_seq_str = "".join(str(int(x)) for x in label_seq)

        return {
            "file": excel_path.name,
            "wid": wid,
            "start": st,
            "end": ed,
            "EU1": EU1s, "EU2": EU2s, "MI1": MI1s, "MI2": MI2s, "GRAY1": G1s, "GRAY2": G2s,
            "EU_folder": eu_folder,
            "MI_folder": mi_folder,
            "GRAY_folder": gray_folder,
            "mode": mode,
            "true_window_label": true_window_label,
            "true_label_seq": true_seq_str,
            "pred_label_seq": pred_seq_str,
            "pred_is_anomaly": int(label_int),
            "pred_label": "anomaly" if label_int else "normal",
            "root_cause_variables": ",".join(root_causes) if root_causes else "",
            "elapsed_sec": float(elapsed),
            "raw": raw,
        }

    if to_process:
        print(f"[INFO] Need VLM for {len(to_process)} windows, max_workers=3 ...")
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(_run_one_window, info) for info in to_process]
            for fut in as_completed(futs):
                win_records.append(fut.result())

    win_records = sorted(win_records, key=lambda r: r["wid"])

    # if mode == "rel_only":
    #     out_xlsx = out_dir / f"window_rel_only__{tag}.xlsx"
    # elif mode == "gray_only":
    #     out_xlsx = out_dir / f"window_gray_only__{tag}.xlsx"
    # else:
    #     out_xlsx = out_dir / f"window_gray__{tag}.xlsx"
    if mode == "gray_only":
        out_xlsx = out_dir / f"window_gray_only__{tag}.xlsx"
    elif mode == "gray_plus_eu":
        out_xlsx = out_dir / f"window_gray_plus_eu__{tag}.xlsx"
    elif mode == "gray_plus_mi":
        out_xlsx = out_dir / f"window_gray_plus_mi__{tag}.xlsx"
    elif mode == "eu_only":
        out_xlsx = out_dir / f"window_eu_only__{tag}.xlsx"
    elif mode == "mi_only":
        out_xlsx = out_dir / f"window_mi_only__{tag}.xlsx"
    elif mode == "eu_plus_mi":
        out_xlsx = out_dir / f"window_eu_plus_mi__{tag}.xlsx"
    elif mode == "value_only":
        out_xlsx = out_dir / f"window_value_only__{tag}.xlsx"
    else:
        out_xlsx = out_dir / f"window_gray__{tag}.xlsx"


    pd.DataFrame(win_records).to_excel(out_xlsx, index=False)
    print(f"[OK] Saved → {out_xlsx}")


def parse_args():
    ap = argparse.ArgumentParser(prog="main.py")

    ap.add_argument("--EU1", type=float, default=0.00, help="EU lower threshold (default: 0.50)")
    ap.add_argument("--EU2", type=float, default=1.00, help="EU upper threshold (default: 0.55)")
    ap.add_argument("--MI1", type=float, default=0.00, help="MI lower threshold (default: 0.80)")
    ap.add_argument("--MI2", type=float, default=1.00, help="MI upper threshold (default: 0.85)")

    ap.add_argument("--GRAY1", type=float, default=0.25, help="GRAY lower threshold (default: 0.30)")
    ap.add_argument("--GRAY2", type=float, default=0.30, help="GRAY upper threshold (default: 0.50)")

    ap.add_argument("--run_gray_plus_rel", type=int, default=0, choices=[0, 1])
    # ap.add_argument("--run_rel_only", type=int, default=0, choices=[0, 1])
    ap.add_argument("--run_gray_plus_eu", type=int, default=0, choices=[0, 1])
    ap.add_argument("--run_gray_plus_mi", type=int, default=0, choices=[0, 1])
    ap.add_argument("--run_gray_only", type=int, default=0, choices=[0, 1])
    ap.add_argument("--run_eu_only", type=int, default=0, choices=[0, 1])
    ap.add_argument("--run_mi_only", type=int, default=0, choices=[0, 1])
    ap.add_argument("--run_eu_plus_mi", type=int, default=1, choices=[0, 1])
    ap.add_argument("--run_value_only", type=int, default=0, choices=[0, 1])

    return ap.parse_args()


def main():
    args = parse_args()
    print(f"run_gray_plus_rel:{args.run_gray_plus_rel}")
    print(f"run_gray_plus_eu:{args.run_gray_plus_eu}")
    print(f"run_gray_plus_mi:{args.run_gray_plus_mi}")
    print(f"run_gray_only:{args.run_gray_only}")
    print(f"run_eu_only:{args.run_eu_only}")
    print(f"run_mi_only:{args.run_mi_only}")
    print(f"run_eu_plus_mi:{args.run_eu_plus_mi}")
    print(f"run_value_only:{args.run_value_only}")
    EU1s, EU2s, MI1s, MI2s = fmt2(args.EU1), fmt2(args.EU2), fmt2(args.MI1), fmt2(args.MI2)
    G1s, G2s = fmt2(args.GRAY1), fmt2(args.GRAY2)

    eu_folder = f"soft_threshold_EU_{EU1s}_{EU2s}_images"
    mi_folder = f"soft_threshold_MI_{MI1s}_{MI2s}_images"
    gray_folder = f"soft_threshold_GRAY_{G1s}_{G2s}_images"

    OUTPUT_DIR.mkdir(exist_ok=True)
    test_files = [p for p in DATA_DIR.glob("*.xlsx") if p.name != "train_data.xlsx"]

    # main 这里仍然保留 priors 加载（你原来的）
    train_data, train_labels, train_seg = load_excel_with_boundary(TRAIN_PATH)
    df_train = pd.read_excel(TRAIN_PATH)
    feature_names = df_train.columns[:NUM_VARS].tolist()

    if TRAIN_PRIORS_PATH.exists():
        print(f"[INFO] Loading train priors from {TRAIN_PRIORS_PATH} ...")
        data_npz = np.load(TRAIN_PRIORS_PATH, allow_pickle=True)
        var_order = data_npz["var_order"].tolist()
        feature_names = data_npz["feature_names"].tolist()
        rgb_db = list(data_npz["rgb_db"])
        print("[INFO] Train priors loaded from cache.")
    else:
        # 仍可按旧逻辑 build，但注意：图现在由 save_extra_images 生成
        print("[INFO] No cache found. Building priors ...")
        all_test_windows_ext = []
        for f in test_files:
            d, _, seg = load_excel_with_boundary(f)
            win_ext_list, _ = sliding_windows_with_boundary(d, seg, WINDOW_SIZE, stride=WINDOW_STRIDE)
            all_test_windows_ext.extend(win_ext_list)

        # global min/max（值/差分/方差）仍按旧逻辑算
        all_test_data = []
        all_test_seg = []
        for f in test_files:
            d, _, seg = load_excel_with_boundary(f)
            all_test_data.append(d)
            all_test_seg.append(seg)
        if all_test_data:
            all_test_data = np.concatenate(all_test_data, axis=1)
            all_test_seg = np.concatenate(all_test_seg, axis=0)
        else:
            all_test_data = np.zeros_like(train_data)
            all_test_seg = np.zeros_like(train_seg)

        concat_data = np.concatenate([train_data, all_test_data], axis=1)
        concat_seg = np.concatenate([train_seg, all_test_seg], axis=0)

        min_vals, max_vals = compute_minmax_stats(concat_data)
        diff_min_vals, diff_max_vals, var_min_vals, var_max_vals = compute_diff_var_minmax_stats(concat_data, concat_seg)
# var_order,
#         min_vals,
#         max_vals,
#         diff_min_vals,
#         diff_max_vals,
#         var_min_vals,
#         var_max_vals,
#         feature_names,
#         rgb_db,
#         eu_val_min, eu_val_max, eu_diff_min, eu_diff_max, eu_var_min, eu_var_max,
#         mi_val_min, mi_val_max, mi_diff_min, mi_diff_max, mi_var_min, mi_var_max,
        (
            var_order,
            _,_,_,_,_,_,
            feature_names,
            rgb_db,
            *__
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
            feature_names=np.array(feature_names),
            rgb_db=np.stack(rgb_db, axis=0),
        )
        print(f"[INFO] Train priors saved to {TRAIN_PRIORS_PATH}")

    # Phase: Inference
    print("[INFO] Inference (assume images were generated by save_extra_images.py)...")
    for f in sorted(test_files):
        out_dir = OUTPUT_DIR / f.stem

        meta_path = out_dir / "window_meta.json"
        if not meta_path.exists():
            print(f"[WARN] Missing window_meta.json for {f.name} -> run save_extra_images.py first. Skip.")
            continue

        eu_dir = out_dir / eu_folder
        mi_dir = out_dir / mi_folder
        g_dir = out_dir / gray_folder

        if not eu_dir.is_dir() or not mi_dir.is_dir() or not g_dir.is_dir():
            print(f"[WARN] Missing folders for {f.name}:")
            print(f"       EU:   {eu_dir}")
            print(f"       MI:   {mi_dir}")
            print(f"       GRAY: {g_dir}")
            print("       Run save_extra_images.py first with matching thresholds. Skip.")
            continue

        tag = f"EU={EU1s}-{EU2s}__MI={MI1s}-{MI2s}__G={G1s}-{G2s}"

        if args.run_gray_plus_rel:
            out_xlsx = out_dir / f"window_gray__{tag}.xlsx"
            if not out_xlsx.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    gray_folder=gray_folder,
                    EU1=args.EU1, EU2=args.EU2, MI1=args.MI1, MI2=args.MI2,
                    GRAY1=args.GRAY1, GRAY2=args.GRAY2,
                    mode="gray_plus_rel",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx}")

        # if args.run_rel_only:
        #     out_xlsx2 = out_dir / f"window_rel_only__{tag}.xlsx"
        #     if not out_xlsx2.exists():
        #         process_file_one_setting(
        #             excel_path=f,
        #             eu_folder=eu_folder,
        #             mi_folder=mi_folder,
        #             gray_folder=gray_folder,
        #             EU1=args.EU1, EU2=args.EU2, MI1=args.MI1, MI2=args.MI2,
        #             GRAY1=args.GRAY1, GRAY2=args.GRAY2,
        #             mode="rel_only",
        #         )
        #     else:
        #         print(f"[INFO] Skip infer: exists {out_xlsx2}")

        if args.run_gray_only:
            out_xlsx2 = out_dir / f"window_gray_only__{tag}.xlsx"
            if not out_xlsx2.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    gray_folder=gray_folder,
                    EU1=args.EU1, EU2=args.EU2, MI1=args.MI1, MI2=args.MI2,
                    GRAY1=args.GRAY1, GRAY2=args.GRAY2,
                    mode="gray_only",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx2}")
        if args.run_gray_plus_eu:
            out_xlsx3 = out_dir / f"window_gray_plus_eu__{tag}.xlsx"
            if not out_xlsx3.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    gray_folder=gray_folder,
                    EU1=args.EU1, EU2=args.EU2, MI1=args.MI1, MI2=args.MI2,
                    GRAY1=args.GRAY1, GRAY2=args.GRAY2,
                    mode="gray_plus_eu",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx3}")
        if args.run_gray_plus_mi:
            out_xlsx4 = out_dir / f"window_gray_plus_mi__{tag}.xlsx"
            if not out_xlsx4.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    gray_folder=gray_folder,
                    EU1=args.EU1, EU2=args.EU2, MI1=args.MI1, MI2=args.MI2,
                    GRAY1=args.GRAY1, GRAY2=args.GRAY2,
                    mode="gray_plus_mi",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx4}")
        if args.run_eu_only:
            out_xlsx5 = out_dir / f"window_eu_only__{tag}.xlsx"
            if not out_xlsx5.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    gray_folder=gray_folder,
                    EU1=args.EU1, EU2=args.EU2, MI1=args.MI1, MI2=args.MI2,
                    GRAY1=args.GRAY1, GRAY2=args.GRAY2,
                    mode="eu_only",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx5}")
        if args.run_mi_only:
            out_xlsx6 = out_dir / f"window_mi_only__{tag}.xlsx"
            if not out_xlsx6.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    gray_folder=gray_folder,
                    EU1=args.EU1, EU2=args.EU2, MI1=args.MI1, MI2=args.MI2,
                    GRAY1=args.GRAY1, GRAY2=args.GRAY2,
                    mode="mi_only",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx6}")    
        if args.run_eu_plus_mi:
            out_xlsx7 = out_dir / f"window_eu_plus_mi__{tag}.xlsx"
            if not out_xlsx7.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    gray_folder=gray_folder,
                    EU1=args.EU1, EU2=args.EU2, MI1=args.MI1, MI2=args.MI2,
                    GRAY1=args.GRAY1, GRAY2=args.GRAY2,
                    mode="eu_plus_mi",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx7}")  
        if args.run_value_only:
            out_xlsx8 = out_dir / f"window_value_only__{tag}.xlsx"
            if not out_xlsx8.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    gray_folder=gray_folder,
                    EU1=args.EU1, EU2=args.EU2, MI1=args.MI1, MI2=args.MI2,
                    GRAY1=args.GRAY1, GRAY2=args.GRAY2,
                    mode="value_only",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx8}")  

if __name__ == "__main__":
    main()
