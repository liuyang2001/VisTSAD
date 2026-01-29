import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent)) 
from src.math_utils import get_6_matrices
from src.utils_io import read_data_file

DATA_DIR = Path("data/dataset/ATSADBench") 
OUTPUT_DIR = Path("data/resources_atsad") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 32 
WINDOW_STRIDE = 32
NUM_VARS = 27    

def get_segments(df):
    segments = []
    start_idx = 0
    df = df.reset_index(drop=True)
    if 'Segment_boundary' not in df.columns:
        if len(df) >= WINDOW_SIZE: segments.append(df)
        return segments
    boundary_indices = df.index[df['Segment_boundary'] == 1].tolist()
    for b_idx in boundary_indices:
        seg = df.loc[start_idx : b_idx]
        if len(seg) >= WINDOW_SIZE: segments.append(seg)
        start_idx = b_idx + 1
    if start_idx < len(df):
        seg = df.loc[start_idx:]
        if len(seg) >= WINDOW_SIZE: segments.append(seg)
    return segments

def main():
    print(">>> Building ATSAD Reference Library (Global Normalization)...")
    
    train_file = DATA_DIR / "train_data.xlsx"
    if not train_file.exists():
        print(f"Error: {train_file} not found!")
        return

    test_files = sorted(list(DATA_DIR.glob("M_*.xlsx")))
    all_files = [train_file] + test_files
    
    print("   [Step 1/3] Calculating Global Raw Stats...")
    all_raw = []
    for f in all_files:
        try:
            df = read_data_file(f)
            vals = df.iloc[:, :NUM_VARS].values
            all_raw.append(vals)
        except Exception as e:
            print(f"Warn: {e}")

    combined = np.vstack(all_raw)
    global_min = combined.min(axis=0)
    global_max = combined.max(axis=0)
    global_rng = global_max - global_min + 1e-12

    print("   [Step 2/3] Calculating Global Matrix Stats (All Files)...")
    matrix_collector = {k: [] for k in ["eu_val", "eu_diff", "eu_var", "mi_val", "mi_diff", "mi_var"]}

    for f in tqdm(all_files, desc="Scanning Matrices"):
        try:
            df = read_data_file(f)
            segments = get_segments(df)
            for seg_df in segments:
                seg_vals = seg_df.iloc[:, :NUM_VARS].values
                seg_norm = ((seg_vals - global_min) / global_rng).T
                
                for s in range(0, seg_norm.shape[1] - WINDOW_SIZE + 1, WINDOW_STRIDE):
                    win_data = seg_norm[:, s:s+WINDOW_SIZE]
                    mats = get_6_matrices(win_data)
                    for k in mats:
                        matrix_collector[k].append(mats[k])
        except:
            pass

    mat_stats = {}
    for k in matrix_collector:
        stack = np.array(matrix_collector[k])
        mn = stack.min(axis=0)
        mx = stack.max(axis=0)
        mat_stats[k] = {"min": mn, "rng": mx - mn + 1e-12}
    
    del matrix_collector

    print("   [Step 3/3] Building Train Library...")
    df_train = read_data_file(train_file)
    train_lib = []
    
    segments = get_segments(df_train)
    for seg_df in segments:
        seg_vals = seg_df.iloc[:, :NUM_VARS].values
        seg_norm = ((seg_vals - global_min) / global_rng).T
        for s in range(0, seg_norm.shape[1] - WINDOW_SIZE + 1, WINDOW_STRIDE):
            win_data = seg_norm[:, s:s+WINDOW_SIZE]
            mats = get_6_matrices(win_data) 
            
            for k in mats:
                mats[k] = (mats[k] - mat_stats[k]["min"]) / mat_stats[k]["rng"]
            
            train_lib.append({"raw_seq": win_data, "mats": mats})

    output_data = {
        "global_stats": {"min": global_min, "rng": global_rng},
        "mat_stats": mat_stats,
        "train_lib": train_lib
    }
    
    save_path = OUTPUT_DIR / "train_ref_lib.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(output_data, f)
        
    print(f"   [Done] ATSAD Library saved to {save_path}")

if __name__ == "__main__":
    main()