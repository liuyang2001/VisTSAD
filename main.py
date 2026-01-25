# main.py
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from generate_images import build_priors
from config import (
    DATA_DIR,
    TRAIN_PATH,
    OUTPUT_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    NUM_VARS,
    TRAIN_LIB_DIR,
    run_all,
    run_NoGeoDis,
    run_NoResGraph,
    run_NoTimefeat,
    run_NoVarCorr,
    EU1,
    EU2,
    MI1,
    MI2
)

from data_utils import (
    load_excel_with_boundary,
)

from vlm_utils import (
    build_direct_prompt_all,
    build_direct_prompt_NoVarCorr,
    build_direct_prompt_NoGeoDis,
    build_direct_prompt_NoTimefeat,
    call_vlm_all,
    call_vlm_NoVarCorr,
    call_vlm_NoGeoDis,
    call_vlm_NoTimefeat,
    call_vlm_NoResGraph
)


TRAIN_PRIORS_PATH = TRAIN_LIB_DIR / "train_priors.npz"


def fmt2(x: float) -> str:
    return f"{x:.2f}"


def get_relation_image_paths(out_dir: Path, wid: int, eu_folder: str, mi_folder: str) -> Dict[str, Path]:
    eu_dir = out_dir / eu_folder
    mi_dir = out_dir / mi_folder
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
    return {
        "eu_val": eu_dir / f"w_{wid:05d}_eu_diff_value_test.png",
        "eu_diff": eu_dir / f"w_{wid:05d}_eu_diff_diff_test.png",
        "eu_var": eu_dir / f"w_{wid:05d}_eu_diff_var_test.png",
    }
def get_mi_relation_image_paths(out_dir: Path, wid: int, eu_folder: str, mi_folder: str) -> Dict[str, Path]:
    eu_dir = out_dir / eu_folder
    mi_dir = out_dir / mi_folder
    return {
        "mi_val": mi_dir / f"w_{wid:05d}_mi_diff_value_test.png",
        "mi_diff": mi_dir / f"w_{wid:05d}_mi_diff_diff_test.png",
        "mi_var": mi_dir / f"w_{wid:05d}_mi_diff_var_test.png",
    }

def process_file_one_setting(
    excel_path: Path,
    eu_folder: str,
    mi_folder: str,
    EU1: float,
    EU2: float,
    MI1: float,
    MI2: float,
    mode: str = "all",
):
    assert mode in ("NoVarCorr","NoGeoDis","all","NoTimefeat")
    EU1s, EU2s, MI1s, MI2s = fmt2(EU1), fmt2(EU2), fmt2(MI1), fmt2(MI2)

    tag = f"EU={EU1s}-{EU2s}__MI={MI1s}-{MI2s}"
    print(f"[INFO] Processing file: {excel_path.name} | {tag} | mode={mode}")

    data, labels, seg = load_excel_with_boundary(excel_path)
    out_dir = OUTPUT_DIR / excel_path.stem

    if mode=="all":
        json_dir = out_dir / f"vlm_all_json__{tag}"
    elif mode == "NoVarCorr":
        json_dir = out_dir / f"vlm_NoVarCorr_json__{tag}"
    elif mode == "NoGeoDis":
        json_dir = out_dir / f"vlm_NoGeoDis_json__{tag}"
    elif mode == "NoTimefeat":
        json_dir = out_dir / f"vlm_NoTimefeat_json__{tag}"
    elif mode == "NoResGraph":
        json_dir = out_dir / f"vlm_NoResGraph_json__{tag}"
    json_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "window_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[ERROR] window_meta.json not found for {excel_path.name}. "
            f"Run generate_images.py first to generate images+meta."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    window_infos: List[Dict[str, Any]] = sorted(meta.get("windows", []), key=lambda x: x["wid"])

    if mode == "all":
        prompt = build_direct_prompt_all()          
    elif mode == "NoVarCorr":
        prompt = build_direct_prompt_NoVarCorr()
    elif mode == "NoGeoDis":
        prompt = build_direct_prompt_NoGeoDis()
    elif mode == "NoTimefeat":
        prompt = build_direct_prompt_NoTimefeat()
    elif mode == "NoResGraph":
        prompt = build_direct_prompt_NoResGraph()


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
                        "EU1": EU1s, "EU2": EU2s, "MI1": MI1s, "MI2": MI2s,
                        "EU_folder": eu_folder,
                        "MI_folder": mi_folder,
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
        eu_paths = get_eu_relation_image_paths(out_dir, wid, eu_folder, mi_folder)
        mi_paths = get_mi_relation_image_paths(out_dir, wid, eu_folder, mi_folder)

        if mode in ("NoVarCorr","all","NoTimefeat"):
            for k, p in eu_paths.items():
                if not p.exists():
                    raise FileNotFoundError(f"[ERROR] Missing eu relation map: {p} (key={k})")
        if mode in ("NoGeoDis","all","NoTimefeat"):
            for k, p in mi_paths.items():
                if not p.exists():
                    raise FileNotFoundError(f"[ERROR] Missing mi relation map: {p} (key={k})")

        t0 = time.perf_counter()


        if mode == "all":
            label_int_tmp, root_causes_tmp, raw = call_vlm_all(
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
                eu_folder=eu_folder, mi_folder=mi_folder,  
            )
        elif mode == "NoVarCorr":
            label_int_tmp, root_causes_tmp, raw = call_vlm_NoVarCorr(
                prompt=prompt,
                img_eu_diff_val_test=rel_paths["eu_val"],
                img_eu_diff_diff_test=rel_paths["eu_diff"],
                img_eu_diff_var_test=rel_paths["eu_var"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,  
            )
        elif mode == "NoGeoDis":
            label_int_tmp, root_causes_tmp, raw = call_vlm_NoGeoDis(
                prompt=prompt,
                img_mi_diff_val_test=rel_paths["mi_val"],
                img_mi_diff_diff_test=rel_paths["mi_diff"],
                img_mi_diff_var_test=rel_paths["mi_var"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,  
            )
        elif mode == "NoTimefeat":
            label_int_tmp, root_causes_tmp, raw = call_vlm_NoTimefeat(
                prompt=prompt,
                img_eu_diff_val_test=rel_paths["eu_val"],
                img_mi_diff_val_test=rel_paths["mi_val"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,  
            )
        elif mode == "NoResGraph":
            label_int_tmp, root_causes_tmp, raw = call_vlm_NoResGraph(
                prompt=prompt,
                img_eu_diff_val_test=rel_paths["eu_val"],
                img_mi_diff_val_test=rel_paths["mi_val"],
                save_dir=json_dir, window_id=wid,
                normal_window_idx=int(info["best_idx"]),
                normal_window_name=str(info["base_name_norm"]),
                eu_folder=eu_folder, mi_folder=mi_folder,  
            )

        elapsed = time.perf_counter() - t0

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
            "EU1": EU1s, "EU2": EU2s, "MI1": MI1s, "MI2": MI2s,
            "EU_folder": eu_folder,
            "MI_folder": mi_folder,
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

    if mode == "all":
        out_xlsx = out_dir / f"window_all__{tag}.xlsx"
    elif mode == "NoVarCorr":
        out_xlsx = out_dir / f"window_NoVarCorr__{tag}.xlsx"
    elif mode == "NoGeoDis":
        out_xlsx = out_dir / f"window_NoGeoDis__{tag}.xlsx"
    elif mode == "NoTimefeat:":
        out_xlsx = out_dir / f"window_NoTimefeat:__{tag}.xlsx"

    pd.DataFrame(win_records).to_excel(out_xlsx, index=False)
    print(f"[OK] Saved → {out_xlsx}")


def main():
    print(f"run_NoVarCorr:{run_NoVarCorr}")
    print(f"run_NoGeoDis:{run_NoGeoDis}")
    print(f"run_all:{run_all}")
    print(f"run_NoTimefeat:{run_NoTimefeat}")
    print(f"run_NoResGraph:{run_NoResGraph}")
    EU1s, EU2s, MI1s, MI2s = fmt2(EU1), fmt2(EU2), fmt2(MI1), fmt2(MI2)

    eu_folder = f"soft_threshold_EU_{EU1s}_{EU2s}_images"
    mi_folder = f"soft_threshold_MI_{MI1s}_{MI2s}_images"

    OUTPUT_DIR.mkdir(exist_ok=True)
    test_files = [p for p in DATA_DIR.glob("*.xlsx") if p.name != "train_data.xlsx"]

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
        print("[INFO] No cache found. Please run generate_images.py first.")
        return
        

    # Phase: Inference
    print("[INFO] Inference (assume images were generated by generate_images.py)...")
    for f in sorted(test_files):
        out_dir = OUTPUT_DIR / f.stem

        meta_path = out_dir / "window_meta.json"
        if not meta_path.exists():
            print(f"[WARN] Missing window_meta.json for {f.name} -> run generate_images.py first. Skip.")
            continue

        eu_dir = out_dir / eu_folder
        mi_dir = out_dir / mi_folder

        if not eu_dir.is_dir() or not mi_dir.is_dir():
            print(f"[WARN] Missing folders for {f.name}:")
            print(f"       EU:   {eu_dir}")
            print(f"       MI:   {mi_dir}")
            print("       Run generate_images.py first with matching thresholds. Skip.")
            continue

        tag = f"EU={EU1s}-{EU2s}__MI={MI1s}-{MI2s}"
        if run_all:
            out_xlsx = out_dir / f"window_all__{tag}.xlsx"
            if not out_xlsx.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    EU1=EU1, EU2=EU2, MI1=MI1, MI2=MI2,
                    mode="all",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx}")  
        elif run_NoVarCorr:
            out_xlsx = out_dir / f"window_NoVarCorr__{tag}.xlsx"
            if not out_xlsx.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    EU1=EU1, EU2=EU2, MI1=MI1, MI2=MI2,
                    mode="NoVarCorr",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx}")
        elif run_NoGeoDis:
            out_xlsx = out_dir / f"window_NoGeoDis__{tag}.xlsx"
            if not out_xlsx.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    EU1=EU1, EU2=EU2, MI1=MI1, MI2=MI2,
                    mode="NoGeoDis",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx}")    
        elif run_NoTimefeat:
            out_xlsx = out_dir / f"window_NoTimefeat__{tag}.xlsx"
            if not out_xlsx.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    EU1=EU1, EU2=EU2, MI1=MI1, MI2=MI2,
                    mode="NoTimefeat",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx}")  
        elif run_NoResGraph:
            out_xlsx = out_dir / f"window_NoResGraph:__{tag}.xlsx"
            if not out_xlsx.exists():
                process_file_one_setting(
                    excel_path=f,
                    eu_folder=eu_folder,
                    mi_folder=mi_folder,
                    EU1=EU1, EU2=EU2, MI1=MI1, MI2=MI2,
                    mode="NoResGraph",
                )
            else:
                print(f"[INFO] Skip infer: exists {out_xlsx}")  

if __name__ == "__main__":
    main()
