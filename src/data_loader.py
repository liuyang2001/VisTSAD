import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir, window_size=64, stride=64):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.num_vars = 8  

    def load_and_slice(self, filename):
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path, sep=None, engine='python')
        
        raw_values = df.iloc[:, :self.num_vars].values
        labels = df['Label'].values
        
        n_samples = len(df)
        num_windows = (n_samples - self.window_size) // self.stride + 1

        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            if end_idx <= n_samples:
                yield {
                    "window_index": i,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "data": raw_values[start_idx:end_idx],  # (64, 8) 
                    "ground_truth": labels[start_idx:end_idx] # (64,) 
                }