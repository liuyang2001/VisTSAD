# metrics.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from config import DATA_DIR, OUTPUT_DIR


def compute_binary_metrics(gt: np.ndarray, pred: np.ndarray):

    gt = gt.astype(int)
    pred = pred.astype(int)

    tp = int(np.sum((gt == 1) & (pred == 1)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    tn = int(np.sum((gt == 0) & (pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return precision, recall, f1, acc, tp, fp, fn, tn


def point_adjustment(gt: np.ndarray, pred: np.ndarray):

    gt = gt.astype(int)
    pred_pa = pred.copy().astype(int)

    anomaly_state = False
    n = len(gt)

    for i in range(n):
        if gt[i] == 1 and pred_pa[i] == 1 and not anomaly_state:
            anomaly_state = True

            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred_pa[j] == 0:
                        pred_pa[j] = 1

            for j in range(i, n):
                if gt[j] == 0:
                    break
                else:
                    if pred_pa[j] == 0:
                        pred_pa[j] = 1

        elif gt[i] == 0:
            anomaly_state = False

        if anomaly_state:
            pred_pa[i] = 1

    pred_pa = np.array(pred_pa, dtype=int)

    accuracy_pa = accuracy_score(gt, pred_pa)
    precision_pa, recall_pa, f1_pa, _ = precision_recall_fscore_support(
        gt, pred_pa, average="binary", zero_division=0
    )

    return pred_pa, precision_pa, recall_pa, f1_pa, accuracy_pa


def rebuild_point_series_from_windows(df: pd.DataFrame):

    max_end = int(df["end"].max())
    T = max_end + 1

    gt = np.zeros(T, dtype=int)
    pred = np.zeros(T, dtype=int)

    for _, row in df.iterrows():
        st = int(row["start"])
        ed = int(row["end"])

        true_seq_str = str(row["true_label_seq"])
        pred_seq_str = str(row["pred_label_seq"])

        gt_seq = np.fromiter(true_seq_str, dtype=int)
        pred_seq = np.fromiter(pred_seq_str, dtype=int)

        length = min(ed - st + 1, len(gt_seq), len(pred_seq))
        if length <= 0:
            continue

        gt[st: st + length] = np.maximum(gt[st: st + length], gt_seq[:length])
        pred[st: st + length] = np.maximum(
            pred[st: st + length], pred_seq[:length]
        )

    return gt, pred


def compute_latency_contiguity(y_true_win: np.ndarray, y_pred_win: np.ndarray):

    y_true_win = y_true_win.astype(int)
    y_pred_win = y_pred_win.astype(int)

    idx_pos = np.where(y_true_win == 1)[0]
    if len(idx_pos) == 0:
        return 0.0, 0.0

    segments = []
    start = idx_pos[0]
    prev = idx_pos[0]
    for idx in idx_pos[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev))
            start = idx
            prev = idx
    segments.append((start, prev))

    latencies = []
    contigs = []

    for (s, e) in segments:
        # --- Latency ---
        first_pred = None
        for j in range(s, len(y_pred_win)):
            if y_pred_win[j] == 1:
                first_pred = j
                break
        if first_pred is None:
            latency = e - s + 1  
        else:
            latency = max(0, first_pred - s)
        latencies.append(latency)

        # --- Contiguity ---
        longest = 0
        cur = 0
        for j in range(s, e + 1):
            if y_pred_win[j] == 1:
                cur += 1
                if cur > longest:
                    longest = cur
            else:
                cur = 0
        seg_len = e - s + 1
        contigs.append(longest / seg_len if seg_len > 0 else 0.0)

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    avg_contiguity = float(np.mean(contigs)) if contigs else 0.0

    return avg_latency, avg_contiguity
def _parse_root_cause_list(val):

    if pd.isna(val):
        return []
    return [s.strip() for s in str(val).split(",") if s.strip()]

def eval_single_dataset(excel_path: Path):

    out_dir = OUTPUT_DIR / excel_path.stem
    win_xlsx = out_dir / "window_EU=0.45-0.50__MI=0.75-0.80.xlsx"
    metrics_xlsx = out_dir / "metrics_EU=0.45-0.50__MI=0.75-0.80.xlsx"

    if not out_dir.exists() or not win_xlsx.exists():
        print(f"[WARN] Skip {excel_path.name}: {win_xlsx} not found.")
        return None

    print(f"\n[INFO] Evaluating dataset: {excel_path.name}")
    df = pd.read_excel(win_xlsx)

    # ========= 1) point-level =========
    gt, pred = rebuild_point_series_from_windows(df)

    precision, recall, f1, accuracy, tp, fp, fn, tn = compute_binary_metrics(
        gt, pred
    )

    pred_pa, precision_pa, recall_pa, f1_pa, accuracy_pa = point_adjustment(
        gt, pred
    )

    # ========= 2) window-level =========
    y_true_win = df["true_window_label"].astype(int).to_numpy()
    y_pred_win = df["pred_is_anomaly"].astype(int).to_numpy()

    (
        precision_win,
        recall_win,
        f1_win,
        accuracy_win,
        tp_w,
        fp_w,
        fn_w,
        tn_w,
    ) = compute_binary_metrics(y_true_win, y_pred_win)

    avg_latency, avg_contiguity = compute_latency_contiguity(
        y_true_win, y_pred_win
    )

    print(f"  Point-level:")
    print(
        f"    precision={precision:.4f}, recall={recall:.4f}, "
        f"f1={f1:.4f}, accuracy={accuracy:.4f}"
    )
    print(
        f"  Point-level (PA):"
        f" precision_pa={precision_pa:.4f}, recall_pa={recall_pa:.4f}, "
        f"f1_pa={f1_pa:.4f}, accuracy_pa={accuracy_pa:.4f}"
    )
    print(f"  Window-level:")
    print(
        f"    precision_win={precision_win:.4f}, recall_win={recall_win:.4f}, "
        f"f1_win={f1_win:.4f}, accuracy_win={accuracy_win:.4f}"
    )
    print(
        f"  Window-level extra:"
        f" avg_latency={avg_latency:.4f}, avg_contiguity={avg_contiguity:.4f}"
    )

    metrics_row = {
        "dataset": excel_path.name,  
        # point-level
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        # point-level PA
        "precision_pa": float(precision_pa),
        "recall_pa": float(recall_pa),
        "f1_pa": float(f1_pa),
        "accuracy_pa": float(accuracy_pa),
        "pred_pa_str": "".join(map(str, pred_pa.tolist())),
        # window-level
        "precision_win": float(precision_win),
        "recall_win": float(recall_win),
        "f1_win": float(f1_win),
        "accuracy_win": float(accuracy_win),
        "tp_win": int(tp_w),
        "fp_win": int(fp_w),
        "fn_win": int(fn_w),
        "tn_win": int(tn_w),
        # extra window metrics
        "avg_latency": float(avg_latency),
        "avg_contiguity": float(avg_contiguity),
        # sizes
        "num_points": int(len(gt)),
        "num_windows": int(len(df)),
    }

    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_excel(metrics_xlsx, index=False)
    print(f"[OK] Saved metrics to {metrics_xlsx}")

    return metrics_row


def main():
    excel_files = [
        p for p in DATA_DIR.glob("*.xlsx") if p.name != "train_data.xlsx"
    ]

    if not excel_files:
        print(f"[WARN] No test excel files found under {DATA_DIR}")
        return

    print(f"[INFO] Found {len(excel_files)} test excel files.")

    all_rows = []
    for f in sorted(excel_files):
        row = eval_single_dataset(f)
        if row is not None:
            all_rows.append(row)

    if all_rows:
        all_df = pd.DataFrame(all_rows)
        all_path = OUTPUT_DIR / "metrics_0.45-0.50__MI=0.75-0.80.xlsx"
        all_df.to_excel(all_path, index=False)
        print(f"[OK] Saved global metrics to {all_path}")
    else:
        print("[WARN] No metrics rows generated; global metrics file not created.")


if __name__ == "__main__":
    main()
