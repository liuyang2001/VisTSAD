import pandas as pd
from pathlib import Path

def read_data_file(file_path):

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    if suffix == '.csv':
        return pd.read_csv(file_path, sep=None, engine='python')
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")