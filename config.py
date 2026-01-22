# config.py
from pathlib import Path


DATA_DIR = Path("dataset")
TRAIN_PATH = DATA_DIR / "train_data.xlsx"
OUTPUT_DIR = Path("outputs")

WINDOW_SIZE = 32
WINDOW_STRIDE = 32
ROLLING_VAR_WINDOW = 7
NUM_VARS = 27 

TRAIN_LIB_DIR = OUTPUT_DIR / "train_lib"