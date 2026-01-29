import argparse
import yaml
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_binary_metrics(gt: np.ndarray, pred: np.ndarray):
    gt = gt.astype(int)
    pred = pred.astype(int)
    tp = int(np.sum((gt == 1) & (pred == 1)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    tn = int(np.sum((gt == 0) & (pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return precision, recall, f1, acc, tp, fp, fn, tn

def point_adjustment(gt: np.ndarray, pred: np.ndarray):
    gt = gt.astype(int)
    pred_pa = pred.copy().astype(int)
    anomaly_state = False
    n = len(gt)
    for i in range(n):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0: break
                pred_pa[j] = 1
            for j in range(i, n):
                if gt[j] == 0: break
                pred_pa[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred_pa[i] = 1
    precision_pa, recall_pa, f1_pa, _ = precision_recall_fscore_support(
        gt, pred_pa, average="binary", zero_division=0
    )
    accuracy_pa = accuracy_score(gt, pred_pa)
    return precision_pa, recall_pa, f1_pa, accuracy_pa

def compute_latency_contiguity_on_windows(win_gt: np.ndarray, win_pred: np.ndarray):
    win_gt = win_gt.astype(int)
    win_pred = win_pred.astype(int)
    idx_pos = np.where(win_gt == 1)[0]
    if len(idx_pos) == 0:
        return 0.0, 1.0 
    segments = []
    start = idx_pos[0]
    for i in range(1, len(idx_pos)):
        if idx_pos[i] != idx_pos[i-1] + 1:
            segments.append((start, idx_pos[i-1]))
            start = idx_pos[i]
    segments.append((start, idx_pos[-1]))

    latencies = []
    contigs = []
    for (s, e) in segments:
        seg_len = e - s + 1
        first_p = -1
        for j in range(s, len(win_pred)):
            if win_pred[j] == 1:
                first_p = j
                break
        latency = (first_p - s) if first_p != -1 else seg_len
        latencies.append(max(0, latency))
        hits = np.sum(win_pred[s:e+1])
        contigs.append(hits / seg_len)
    return np.mean(latencies), np.mean(contigs)


def load_config():
    with open("config/main_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def reconstruct_series(result_df, original_df):
    gt_full = original_df['Label'].values.astype(int)
    pred_full = np.zeros_like(gt_full, dtype=int)
    for _, row in result_df.iterrows():
        s = int(row['Start_Time'])
        e = int(row['End_Time'])
        try:
            raw_array_str = row['Raw_Pred_Array']
            local_preds = ast.literal_eval(raw_array_str)
            length_to_fill = min(e, len(pred_full)) - s
            if length_to_fill > 0:
                pred_full[s : s+length_to_fill] = local_preds[:length_to_fill]
        except Exception as ex:
            print(f"Error parsing window at {s}: {ex}")
    return gt_full, pred_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None, help="Override config mode")
    args = parser.parse_args()

    cfg = load_config()
    mode = args.mode if args.mode else cfg['experiment']['mode']
    
    data_dir = Path(cfg['paths']['dataset'])
    base_output_dir = Path(cfg['paths']['output_dir'])
    target_results_dir = base_output_dir / mode 
    
    if not target_results_dir.exists():
        print(f"Error: Results directory {target_results_dir} not found.")
        return

    print(f"=== Evaluating Metrics for Mode: [{mode}] ===")
    
    result_files = sorted(list(target_results_dir.glob(f"Result_*_{mode}.xlsx")))
    if not result_files:
        print(f"No result files found in {target_results_dir}")
        return

    summary_list = []
    
    for res_file in result_files:
        fname_base = res_file.name.replace("Result_", "").replace(f"_{mode}.xlsx", "")
        csv_name = fname_base + ".csv"
        csv_path = data_dir / csv_name
        
        if not csv_path.exists():
            print(f"  [Warn] Original data {csv_name} not found, skipping.")
            continue
            
        print(f"  -> Processing: {fname_base}")
        
        try:
            res_df = pd.read_excel(res_file)
            orig_df = pd.read_csv(csv_path, sep=None, engine='python')
            
            if 'Latency' in res_df.columns:
                avg_inference_time = res_df['Latency'].mean()
            elif 'Latency_Seconds' in res_df.columns:
                avg_inference_time = res_df['Latency_Seconds'].mean()
            else:
                avg_inference_time = 0.0

            gt, pred = reconstruct_series(res_df, orig_df)
            
            # 1. Point Metrics
            p, r, f1, acc, tp, fp, fn, tn = compute_binary_metrics(gt, pred)
            
            # 2. Point Adjustment
            p_pa, r_pa, f1_pa, acc_pa = point_adjustment(gt, pred)
            
            # 3. Window Metrics
            WINDOW_SIZE = 64
            win_gt = []
            win_pred = []
            for i in range(0, len(gt), WINDOW_SIZE):
                chunk_gt = gt[i : i + WINDOW_SIZE]
                chunk_pred = pred[i : i + WINDOW_SIZE]
                win_gt.append(1 if np.any(chunk_gt == 1) else 0)
                win_pred.append(1 if np.any(chunk_pred == 1) else 0)
            win_gt = np.array(win_gt)
            win_pred = np.array(win_pred)
            pw, rw, f1w, accw, tpw, fpw, fnw, tnw = compute_binary_metrics(win_gt, win_pred)
            
            # 4. Latency & Contiguity
            avg_lat, avg_cont = compute_latency_contiguity_on_windows(win_gt, win_pred)
            
            row = {
                "Dataset": fname_base,
                "P_point": round(p, 4), "R_point": round(r, 4), "F1_point": round(f1, 4), "Acc_point": round(acc, 4),
                "P_PA": round(p_pa, 4), "R_PA": round(r_pa, 4), "F1_PA": round(f1_pa, 4), "Acc_PA": round(acc_pa, 4),
                "P_win": round(pw, 4), "R_win": round(rw, 4), "F1_win": round(f1w, 4), "Acc_win": round(accw, 4),
                "Avg_Latency_Win": round(avg_lat, 2),
                "Avg_Contiguity_Win": round(avg_cont, 4),
                "Avg_VLM_Latency": round(avg_inference_time, 4),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn
            }
            summary_list.append(row)
            
            pd.DataFrame([row]).to_excel(res_file.parent / f"Metrics_{fname_base}.xlsx", index=False)

        except Exception as e:
            print(f"    [Error] {fname_base}: {e}")

    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        mean_row = summary_df.mean(numeric_only=True)
        mean_row["Dataset"] = "AVERAGE"
        summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)
        
        summary_path = target_results_dir / f"SUMMARY_METRICS_{mode}.xlsx"
        summary_df.to_excel(summary_path, index=False)
        
        print("\n" + "="*50)
        print(f"FINAL SUMMARY - {mode.upper()}")
        print(f"Avg F1 (Point):      {mean_row['F1_point']:.4f}")
        print(f"Avg F1 (PA):         {mean_row['F1_PA']:.4f}")
        print(f"Avg F1 (Window):     {mean_row['F1_win']:.4f}")
        print(f"Avg Detection Delay: {mean_row['Avg_Latency_Win']:.2f} windows")
        print(f"Avg Inference Time:  {mean_row['Avg_VLM_Latency']:.4f} s") 
        print(f"Saved to: {summary_path}")
        print("="*50)

if __name__ == "__main__":
    main()