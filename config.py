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
MODEL_NAME="qwen3-vl-235b-a22b-thinking"
API_KEY=""
BASE_URL=""

run_all=1
run_NoVarCorr=0
run_NoGeoDis=0
run_NoTimefeat=0
run_NoResGraph=0

EU1=0.45
EU2=0.50
MI1=0.75
MI2=0.80