# vlm_utils.py
from pathlib import Path
import base64
import json
from openai import OpenAI
from config import WINDOW_SIZE,MODEL_NAME,API_KEY,BASE_URL
import time

DIRECT_PROMPT_NoVarCorr = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 3 images describing a TEST window (32 time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the Euclidean distance between every pair of variables based on their 32-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise Euclidean-distance matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the difference from NORMAL.

(2) TEST Euclidean-diff (diff)
    - Same as (1), but the Euclidean distance is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(3) TEST Euclidean-diff (var)
    - Same as (1), but the Euclidean distance is computed on rolling-variance sequences(local volatility over time with window size 7).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the Euclidean-diff images (value, diff, and rolling variance), decide for the 32 time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length 32.
   - IMPORTANT:
       - If the Euclidean-diff images show abnormal patterns, this indicates a global structural anomaly of the window; in this case, you should mark all 32 time steps as anomalous (all 1s).

2. Root Cause Analysis
   - If all time steps are anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The Euclidean-diff maps (1)-(3)

=====================================================
REASONING FORMAT (REQUIRED)
=====================================================

Analysis Process:
- Briefly describe how you used the images to decide the 32-step anomaly labels.
- Briefly describe how you used the images to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {
  "Label": [l_1, l_2, ..., l_32],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}

Rules:
- "Label" MUST be an array of exactly 32 integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""
DIRECT_PROMPT_NoGeoDis = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 3 images describing a TEST window (32 time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the mutual information between every pair of variables based on their 32-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise mutual information matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the difference from NORMAL.

(2) TEST MI-diff (diff)
    - Same as (1), but the mutual information is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(3) TEST MI-diff (var)
    - Same as (1), but the mutual information is computed on rolling-variance sequences(local volatility over time with window size 7).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the MI-diff images (value, diff, and rolling variance), decide for the 32 time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length 32.
   - IMPORTANT:
       - If the MI-diff images show abnormal patterns, this indicates a global structural anomaly of the window; in this case, you should mark all 32 time steps as anomalous (all 1s).

2. Root Cause Analysis
   - If all time steps are anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The MI-diff maps (1)-(3)

=====================================================
REASONING FORMAT (REQUIRED)
=====================================================

Analysis Process:
- Briefly describe how you used the images to decide the 32-step anomaly labels.
- Briefly describe how you used the images to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {
  "Label": [l_1, l_2, ..., l_32],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}

Rules:
- "Label" MUST be an array of exactly 32 integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""
DIRECT_PROMPT_all = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 6 images describing a TEST window (32 time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the Euclidean distance between every pair of variables based on their 32-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise Euclidean-distance matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the difference from NORMAL.

(2) TEST Euclidean-diff (diff)
    - Same as (1), but the Euclidean distance is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(3) TEST Euclidean-diff (var)
    - Same as (1), but the Euclidean distance is computed on rolling-variance sequences(local volatility over time with window size 7).

(4) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the mutual information between every pair of variables based on their 32-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise mutual information matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the difference from NORMAL.

(5) TEST MI-diff (diff)
    - Same as (4), but the mutual information is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(6) TEST MI-diff (var)
    - Same as (4), but the mutual information is computed on rolling-variance sequences(local volatility over time with window size 7).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the Euclidean-diff images and MI-diff images (value, diff, and rolling variance), decide for the 32 time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length 32.
   - IMPORTANT:
       - If the Euclidean-diff images or MI-diff images show abnormal patterns, this indicates a global structural anomaly of the window; in this case, you should mark all 32 time steps as anomalous (all 1s).

2. Root Cause Analysis
   - If all time steps are anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The Euclidean-diff maps (1)-(3)
       - The MI-diff maps (4)-(6)

=====================================================
REASONING FORMAT (REQUIRED)
=====================================================

Analysis Process:
- Briefly describe how you used the images to decide the 32-step anomaly labels.
- Briefly describe how you used the images to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {
  "Label": [l_1, l_2, ..., l_32],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}

Rules:
- "Label" MUST be an array of exactly 32 integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""
DIRECT_PROMPT_NoTimefeat = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 2 images describing a TEST window (32 time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the Euclidean distance between every pair of variables based on their 32-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise Euclidean-distance matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the difference from NORMAL.

(2) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the mutual information between every pair of variables based on their 32-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise mutual information matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the difference from NORMAL.

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the Euclidean-diff image and MI-diff image, decide for the 32 time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length 32.
   - IMPORTANT:
       - If the Euclidean-diff image or MI-diff image shows abnormal patterns, this indicates a global structural anomaly of the window; in this case, you should mark all 32 time steps as anomalous (all 1s).

2. Root Cause Analysis
   - If all time steps are anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The Euclidean-diff map (1)
       - The MI-diff map (2)

=====================================================
REASONING FORMAT (REQUIRED)
=====================================================

Analysis Process:
- Briefly describe how you used the images to decide the 32-step anomaly labels.
- Briefly describe how you used the images to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {
  "Label": [l_1, l_2, ..., l_32],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}

Rules:
- "Label" MUST be an array of exactly 32 integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""

def build_direct_prompt_NoVarCorr() -> str:
    return DIRECT_PROMPT_NoVarCorr
def build_direct_prompt_NoGeoDis() -> str:
    return DIRECT_PROMPT_NoGeoDis
def build_direct_prompt_all() -> str:
    return DIRECT_PROMPT_all
def build_direct_prompt_NoTimefeat() -> str:
    return DIRECT_PROMPT_NoTimefeat

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)
def _to_b64(p: Path) -> str:
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _parse_vlm_json(text: str):
    js = text[text.find("{"): text.rfind("}") + 1]
    obj = json.loads(js)

    raw_label = obj.get("Label", 0)

    if isinstance(raw_label, list):
        tmp = []
        for x in raw_label:
            try:
                v = int(x)
            except Exception:
                v = 0
            tmp.append(1 if v == 1 else 0)

        if len(tmp) >= WINDOW_SIZE:
            label_seq = tmp[:WINDOW_SIZE]
        else:
            label_seq = tmp + [0] * (WINDOW_SIZE - len(tmp))
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

    if label_int == 0:
        root_causes = []

    return label_int, label_seq, root_causes


def call_vlm_NoVarCorr(
    prompt: str,
    img_eu_diff_val_test: Path,
    img_eu_diff_diff_test: Path,
    img_eu_diff_var_test: Path,
    save_dir: Path,
    window_id: int,
    normal_window_idx: int,
    normal_window_name: str,
    eu_folder: str,
    mi_folder: str,
):
    content = [
        {"type": "text", "text": prompt},

        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_var_test)}},
    ]
    print(f"the images input to LLM:")
    print(img_eu_diff_val_test)
    print(img_eu_diff_diff_test)
    print(img_eu_diff_var_test)
    label_int = 0
    label_seq = [0] * WINDOW_SIZE
    root_causes = []
    text = ""
    label_str = "normal"

    # try:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    raw = completion.choices[0].message.content
    text = "".join(x.get("text", str(x)) for x in raw) if isinstance(raw, list) else raw
    try:
        label_int, label_seq, root_causes = _parse_vlm_json(text)
        label_str = "anomaly" if label_int == 1 else "normal"
    except Exception as e:
        print(f"[WARN] JSON parse failed (gray+rel) for window {window_id}: {e}")
        label_int = 0
        label_seq = [0] * WINDOW_SIZE
        root_causes = []
        label_str = "normal"
    # except Exception as e:
    #     print(f"[WARN] VLM API call failed (gray+rel): {e}")
    #     text = f"[ERROR] VLM failed: {e}"
    #     label_int = 0
    #     label_seq = [0] * WINDOW_SIZE
    #     root_causes = []
    #     label_str = "normal"

    save_path = save_dir / f"vlm_reply_w_{window_id:05d}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "Analysis Process": text,
                "Label": [int(x) for x in label_seq],
                "root_cause_variables": root_causes,
                "window_level_label": int(label_int),
                "raw_label": label_str,
                "normal_window_index": int(normal_window_idx),
                "normal_window_name": normal_window_name,
                "eu_folder": eu_folder,
                "mi_folder": mi_folder,
                "mode": "NoVarCorr",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text
def call_vlm_NoGeoDis(
    prompt: str,
    img_mi_diff_val_test: Path,
    img_mi_diff_diff_test: Path,
    img_mi_diff_var_test: Path,
    save_dir: Path,
    window_id: int,
    normal_window_idx: int,
    normal_window_name: str,
    eu_folder: str,
    mi_folder: str,
):
    content = [
        {"type": "text", "text": prompt},

        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_var_test)}},
    ]
    print(f"the images input to LLM:")
    print(img_mi_diff_val_test)
    print(img_mi_diff_diff_test)
    print(img_mi_diff_var_test)
    label_int = 0
    label_seq = [0] * WINDOW_SIZE
    root_causes = []
    text = ""
    label_str = "normal"

    # try:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    raw = completion.choices[0].message.content
    text = "".join(x.get("text", str(x)) for x in raw) if isinstance(raw, list) else raw
    try:
        label_int, label_seq, root_causes = _parse_vlm_json(text)
        label_str = "anomaly" if label_int == 1 else "normal"
    except Exception as e:
        print(f"[WARN] JSON parse failed (mi only) for window {window_id}: {e}")
        label_int = 0
        label_seq = [0] * WINDOW_SIZE
        root_causes = []
        label_str = "normal"
    # except Exception as e:
    #     print(f"[WARN] VLM API call failed (gray+rel): {e}")
    #     text = f"[ERROR] VLM failed: {e}"
    #     label_int = 0
    #     label_seq = [0] * WINDOW_SIZE
    #     root_causes = []
    #     label_str = "normal"

    save_path = save_dir / f"vlm_reply_w_{window_id:05d}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "Analysis Process": text,
                "Label": [int(x) for x in label_seq],
                "root_cause_variables": root_causes,
                "window_level_label": int(label_int),
                "raw_label": label_str,
                "normal_window_index": int(normal_window_idx),
                "normal_window_name": normal_window_name,
                "eu_folder": eu_folder,
                "mi_folder": mi_folder,
                "mode": "NoGeoDis",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text
def call_vlm_all(
    prompt: str,
    img_eu_diff_val_test: Path,
    img_eu_diff_diff_test: Path,
    img_eu_diff_var_test: Path,
    img_mi_diff_val_test: Path,
    img_mi_diff_diff_test: Path,
    img_mi_diff_var_test: Path,
    save_dir: Path,
    window_id: int,
    normal_window_idx: int,
    normal_window_name: str,
    eu_folder: str,
    mi_folder: str,
):
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_var_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_var_test)}},
    ]
    print(f"the images input to LLM:")
    print(img_eu_diff_val_test)
    print(img_eu_diff_diff_test)
    print(img_eu_diff_var_test)
    print(img_mi_diff_val_test)
    print(img_mi_diff_diff_test)
    print(img_mi_diff_var_test)
    label_int = 0
    label_seq = [0] * WINDOW_SIZE
    root_causes = []
    text = ""
    label_str = "normal"

    # try:
    time_s=time.time()
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    time_e=time.time()
    time_sec=time_e-time_s
    raw = completion.choices[0].message.content
    text = "".join(x.get("text", str(x)) for x in raw) if isinstance(raw, list) else raw
    try:
        label_int, label_seq, root_causes = _parse_vlm_json(text)
        label_str = "anomaly" if label_int == 1 else "normal"
    except Exception as e:
        print(f"[WARN] JSON parse failed (eu plus mi) for window {window_id}: {e}")
        label_int = 0
        label_seq = [0] * WINDOW_SIZE
        root_causes = []
        label_str = "normal"
    # except Exception as e:
    #     print(f"[WARN] VLM API call failed (gray+rel): {e}")
    #     text = f"[ERROR] VLM failed: {e}"
    #     label_int = 0
    #     label_seq = [0] * WINDOW_SIZE
    #     root_causes = []
    #     label_str = "normal"

    save_path = save_dir / f"vlm_reply_w_{window_id:05d}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "Analysis Process": text,
                "Label": [int(x) for x in label_seq],
                "root_cause_variables": root_causes,
                "window_level_label": int(label_int),
                "raw_label": label_str,
                "normal_window_index": int(normal_window_idx),
                "normal_window_name": normal_window_name,
                "eu_folder": eu_folder,
                "mi_folder": mi_folder,
                "mode": "ALL",
                "time":time_sec
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text
def call_vlm_NoTimefeat(
    prompt: str,
    img_eu_diff_val_test: Path,
    # img_eu_diff_diff_test: Path,
    # img_eu_diff_var_test: Path,
    img_mi_diff_val_test: Path,
    # img_mi_diff_diff_test: Path,
    # img_mi_diff_var_test: Path,
    save_dir: Path,
    window_id: int,
    normal_window_idx: int,
    normal_window_name: str,
    eu_folder: str,
    mi_folder: str,
):
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_val_test)}},
        # {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_diff_test)}},
        # {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_var_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_val_test)}},
        # {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_diff_test)}},
        # {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_var_test)}},
    ]
    print(f"the images input to LLM:")
    print(img_eu_diff_val_test)
    # print(img_eu_diff_diff_test)
    # print(img_eu_diff_var_test)
    print(img_mi_diff_val_test)
    # print(img_mi_diff_diff_test)
    # print(img_mi_diff_var_test)
    label_int = 0
    label_seq = [0] * WINDOW_SIZE
    root_causes = []
    text = ""
    label_str = "normal"

    # try:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    raw = completion.choices[0].message.content
    text = "".join(x.get("text", str(x)) for x in raw) if isinstance(raw, list) else raw
    try:
        label_int, label_seq, root_causes = _parse_vlm_json(text)
        label_str = "anomaly" if label_int == 1 else "normal"
    except Exception as e:
        print(f"[WARN] JSON parse failed (eu plus mi) for window {window_id}: {e}")
        label_int = 0
        label_seq = [0] * WINDOW_SIZE
        root_causes = []
        label_str = "normal"
    # except Exception as e:
    #     print(f"[WARN] VLM API call failed (gray+rel): {e}")
    #     text = f"[ERROR] VLM failed: {e}"
    #     label_int = 0
    #     label_seq = [0] * WINDOW_SIZE
    #     root_causes = []
    #     label_str = "normal"

    save_path = save_dir / f"vlm_reply_w_{window_id:05d}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "Analysis Process": text,
                "Label": [int(x) for x in label_seq],
                "root_cause_variables": root_causes,
                "window_level_label": int(label_int),
                "raw_label": label_str,
                "normal_window_index": int(normal_window_idx),
                "normal_window_name": normal_window_name,
                "eu_folder": eu_folder,
                "mi_folder": mi_folder,
                "mode": "NoTimefeat",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text
def call_vlm_NoResGraph(
    prompt: str,
    img_eu_diff_val_test: Path,
    img_eu_diff_diff_test: Path,
    img_eu_diff_var_test: Path,
    img_mi_diff_val_test: Path,
    img_mi_diff_diff_test: Path,
    img_mi_diff_var_test: Path,
    save_dir: Path,
    window_id: int,
    normal_window_idx: int,
    normal_window_name: str,
    eu_folder: str,
    mi_folder: str,
):
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_var_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_var_test)}},
    ]
    print(f"the images input to LLM:")
    print(img_eu_diff_val_test)
    print(img_eu_diff_diff_test)
    print(img_eu_diff_var_test)
    print(img_mi_diff_val_test)
    print(img_mi_diff_diff_test)
    print(img_mi_diff_var_test)
    label_int = 0
    label_seq = [0] * WINDOW_SIZE
    root_causes = []
    text = ""
    label_str = "normal"

    # try:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    raw = completion.choices[0].message.content
    text = "".join(x.get("text", str(x)) for x in raw) if isinstance(raw, list) else raw
    try:
        label_int, label_seq, root_causes = _parse_vlm_json(text)
        label_str = "anomaly" if label_int == 1 else "normal"
    except Exception as e:
        print(f"[WARN] JSON parse failed (eu plus mi) for window {window_id}: {e}")
        label_int = 0
        label_seq = [0] * WINDOW_SIZE
        root_causes = []
        label_str = "normal"
    # except Exception as e:
    #     print(f"[WARN] VLM API call failed (gray+rel): {e}")
    #     text = f"[ERROR] VLM failed: {e}"
    #     label_int = 0
    #     label_seq = [0] * WINDOW_SIZE
    #     root_causes = []
    #     label_str = "normal"

    save_path = save_dir / f"vlm_reply_w_{window_id:05d}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "Analysis Process": text,
                "Label": [int(x) for x in label_seq],
                "root_cause_variables": root_causes,
                "window_level_label": int(label_int),
                "raw_label": label_str,
                "normal_window_index": int(normal_window_idx),
                "normal_window_name": normal_window_name,
                "eu_folder": eu_folder,
                "mi_folder": mi_folder,
                "mode": "NoResGraph",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text