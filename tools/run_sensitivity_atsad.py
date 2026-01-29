import sys
import subprocess
import numpy as np
import json
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils_io import read_data_file

try:
    from eval_metrics import reconstruct_series
except ImportError:
    pass

MULTIPLIERS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

BASE_CONFIG_PATH = Path("config/atsad/dataset_params_all.json")

TEMP_CONFIG_DIR = Path("config/atsad_sensitivity_temp")
RESULTS_ROOT = Path("results_atsad/sensitivity")
DATA_DIR = Path("data/dataset/ATSADBench")
MAIN_SCRIPT = "main.py"

TARGET_FILES = [
    "M_IL_CDA_test_data.xlsx", 
    "M_IL_FVA_test_data.xlsx", 
    "M_IL_TVDA_test_data.xlsx",
    "M_OL_CDA_test_data.xlsx", 
    "M_OL_FVA_test_data.xlsx", 
    "M_OL_TVDA_test_data.xlsx"
]

def compute_metrics_special(gt, pred):
    gt = gt.astype(int)
    pred = pred.astype(int)
    tp = int(np.sum((gt == 1) & (pred == 1)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    
    if (tp + fp) == 0: precision = 1.0
    else: precision = tp / (tp + fp)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def generate_temp_config(multiplier, save_path):

    print(f"  -> Generating config for multiplier x{multiplier}...")
    
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_CONFIG_PATH}. Please run tools/generate_atsad_params.py first.")
    
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_config = json.load(f)
    
    new_config = {}
    
    for fname, params in base_config.items():
        new_file_params = {}
        
        for matrix_key, p_vals in params.items():
            base_t1 = p_vals["t1"]
            base_t2 = p_vals["t2"]
            
            if base_t2 < 1.0: 
                new_t1 = base_t1 * multiplier
                new_t2 = base_t2 * multiplier
                
                new_file_params[matrix_key] = {
                    "t1": round(new_t1, 5),
                    "t2": round(new_t2, 5)
                }
            else:
                new_file_params[matrix_key] = {"t1": 1.0, "t2": 1.0}
        
        new_config[fname] = new_file_params

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2)

def run_pipeline():
    summary_rows = []
    
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} does not exist.")
        return

    for m in MULTIPLIERS:
        print(f"\n{'='*60}")
        print(f"ATSAD Sensitivity: Multiplier = {m}")
        print(f"{'='*60}")
        
        run_output_dir = RESULTS_ROOT / f"M_{m:.2f}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        temp_config_path = TEMP_CONFIG_DIR / f"params_x{m:.2f}.json"
        
        generate_temp_config(m, temp_config_path)
        
        print(f"  -> Starting Inference...")
        cmd_main = [
            "python", MAIN_SCRIPT,
            "--config", "config/main_config_atsad.yaml", 
            "--params_file", str(temp_config_path),
            "--output_dir", str(run_output_dir),
            "--mode", "all"
        ]
        
        subprocess.run(cmd_main)
        
        print(f"  -> Calculating Precision/Recall...")
        all_files = list(run_output_dir.glob("Result_*.xlsx"))
        p_list, r_list, f1_list = [], [], []
        
        for res_file in all_files:
            try:
                fname_base = res_file.name.replace("Result_", "").replace("_all.xlsx", "")
                
                real_fname = None
                if (DATA_DIR / f"{fname_base}.xlsx").exists():
                    real_fname = f"{fname_base}.xlsx"
                else:
                    for tf in TARGET_FILES:
                        if Path(tf).stem == fname_base: 
                            real_fname = tf
                            break
                
                if not real_fname: continue

                file_path = DATA_DIR / real_fname
                
                res_df = pd.read_excel(res_file)
                orig_df = read_data_file(file_path)
                
                gt, pred = reconstruct_series(res_df, orig_df)
                p, r, f1 = compute_metrics_special(gt, pred)
                
                p_list.append(p)
                r_list.append(r)
                f1_list.append(f1)
                
            except Exception as e:
                print(f"    [Err] {res_file.name}: {e}")

        avg_p = np.mean(p_list) if p_list else 0.0
        avg_r = np.mean(r_list) if r_list else 0.0
        avg_f1 = np.mean(f1_list) if f1_list else 0.0
        
        print(f"  -> [x{m:.2f}] P: {avg_p:.4f} | R: {avg_r:.4f} | F1: {avg_f1:.4f}")
        summary_rows.append({"Multiplier": m, "Avg_P": avg_p, "Avg_R": avg_r, "Avg_F1": avg_f1})

    final_df = pd.DataFrame(summary_rows)
    final_save_path = RESULTS_ROOT / "FINAL_SUMMARY.xlsx" 
    final_df = final_df.round(4)
    final_df.to_excel(final_save_path, index=False)
    
    print(f"\nATSAD Sensitivity Finished. Summary saved to {final_save_path}")

if __name__ == "__main__":
    run_pipeline()