# VisTASD Project

## 1. Setup and Preparation

**Step 1: Install Dependencies**
Install the required dependencies based on `requirement.txt`:
```bash
pip install -r requirement.txt
```

**Step 2: Dataset Preparation**
Place the corresponding dataset files into the dataset folder. 

**Step 3: Configuration**
Open `config.py` and fill in the following parameters:
- `MODEL_NAME`
- `API_KEY`
- `BASE_URL`

**Step 4: Generate Graphs**
Run the image generation script to obtain the corresponding graphs:
```bash
python generate_images.py
```

---

## 2. Running VisTASD

**Step 1: Run Main Script**
Run `main.py` to generate the results:
```bash
python main.py
```

**Step 2: Get Metrics**
Run `metrics.py` to obtain statistical indicators:
```bash
python metrics.py
```

---

## 3. Ablation Studies (VisTSAD)

**Step 1: Modify Configuration**
Open `config.py` and adjust the settings:
1. Set `run_all = 0`.
2. Set the target ablation experiment flag to `1`. Select one of the following:
   - `run_NoVarCorr`
   - `run_NoGeoDis`
   - `run_NoTimefeat`
   - `run_NoResGraph`

**Step 2: Run Main Script**
Run `main.py` to generate the results for the selected ablation study:
```bash
python main.py
```

**Step 3: Get Metrics**
Run `metrics.py` to obtain statistical indicators:
```bash
python metrics.py
```

## 4. Different Scaling Factors

**Step 1: Modify Configuration**
Open config.py and adjust the settings by multiplying the following parameters by the corresponding scaling factor:
EU1,EU2,MI1,MI2

**Step 2: Run Main Script**
Run `main.py` to generate the results:
```bash
python main.py
```

**Step 3: Get Metrics**
Run `metrics.py` to obtain statistical indicators:
```bash
python metrics.py
```