# vlm_utils.py
from pathlib import Path
import base64
import json
from openai import OpenAI
from config import WINDOW_SIZE
import time
# ============================================================
# Prompt：3张“时序差值图”(value/diff/var 的 TEST-NORMAL, soft-threshold 后 [-1,1] 蓝白红)
#       + 6张变量关系图(EU/MI diff, value/diff/var)
# ============================================================
DIRECT_PROMPT_GRAY_PLUS_REL = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 9 images describing a TEST window (32 time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST-NORMAL time-series-diff map (value)
    - A matrix image: rows = variables, columns = time steps (length 32).
    - First, the value sequences are normalized per variable to [0, 1] for the TEST window.
    - Then the value sequences are normalized per variable to [0, 1] for the matched NORMAL window.
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(2) TEST-NORMAL time-series-diff map (diff)
    - Same format as (1), but computed on first-difference sequences.
    - First, first-difference sequences are computed for the TEST window:
      diff[t] = value[t] - value[t-1].
    - Then the same first-difference sequences are computed for the matched NORMAL window.
    - Then each is normalized per variable to [0, 1].
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0,
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(3) TEST-NORMAL time-series-diff map (rolling variance)
    - Same format as (1), but computed on rolling-variance sequences.
    - First, rolling-variance sequences (window size 7) are computed for the TEST window.
    - Then the same rolling-variance sequences are computed for the matched NORMAL window.
    - Then each is normalized per variable to [0, 1].
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0,
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(4) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, a pairwise Euclidean-distance matrix between value sequences is computed for the TEST window.
    - Then the same pairwise Euclidean-distance matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the difference from NORMAL.

(5) TEST Euclidean-diff (diff)
    - Same as (4), but the Euclidean distance is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(6) TEST Euclidean-diff (var)
    - Same as (4), but the Euclidean distance is computed on rolling-variance sequences(local volatility over time with window size 7).

(7) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, a mutual-information matrix between value sequences is computed for the TEST window.
    - Then the same mutual-information matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.

(8) TEST MI-diff (diff)
    - Same as (7), but mutual information is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(9) TEST MI-diff (var)
    - Same as (7), but mutual information is computed on rolling-variance sequences(local volatility over time with window size 7).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the time-series-diff maps (value, diff, and rolling variance), decide for each of the 32 time steps whether it is normal (0) or anomalous (1).
   - The output should be a binary array of length 32.
   - IMPORTANT:
       - The time-series-diff maps allow you to assess anomalies at individual time steps.
       - If the Euclidean-diff or MI-diff images show abnormal patterns, this indicates a global structural anomaly of the window; in this case, you should mark all 32 time steps as anomalous (all 1s).

2. Root Cause Analysis
   - If any time step is anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The time-series diff maps (1)-(3)
       - The Euclidean-diff maps (4)-(6)
       - The MI-diff maps (7)-(9)

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

# ============================================================
# Prompt：3张“时序差值图”(value/diff/var)
#       + 3张 EU 关系差图(EU diff, value/diff/var)
# ============================================================
DIRECT_PROMPT_GRAY_PLUS_EU = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 6 images describing a TEST window (32 time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST-NORMAL time-series-diff map (value)
    - A matrix image: rows = variables, columns = time steps (length 32).
    - First, the value sequences are normalized per variable to [0, 1] for the TEST window.
    - Then the value sequences are normalized per variable to [0, 1] for the matched NORMAL window.
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(2) TEST-NORMAL time-series-diff map (diff)
    - Same format as (1), but computed on first-difference sequences.
    - First, first-difference sequences are computed for the TEST window:
      diff[t] = value[t] - value[t-1].
    - Then the same first-difference sequences are computed for the matched NORMAL window.
    - Then each is normalized per variable to [0, 1].
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0,
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(3) TEST-NORMAL time-series-diff map (rolling variance)
    - Same format as (1), but computed on rolling-variance sequences.
    - First, rolling-variance sequences (window size 7) are computed for the TEST window.
    - Then the same rolling-variance sequences are computed for the matched NORMAL window.
    - Then each is normalized per variable to [0, 1].
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0,
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(4) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, a pairwise Euclidean-distance matrix between value sequences is computed for the TEST window.
    - Then the same pairwise Euclidean-distance matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the difference from NORMAL.

(5) TEST Euclidean-diff (diff)
    - Same as (4), but the Euclidean distance is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(6) TEST Euclidean-diff (var)
    - Same as (4), but the Euclidean distance is computed on rolling-variance sequences(local volatility over time with window size 7).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the time-series-diff maps (value, diff, and rolling variance), decide for each of the 32 time steps whether it is normal (0) or anomalous (1).
   - The output should be a binary array of length 32.
   - IMPORTANT:
       - The time-series-diff maps allow you to assess anomalies at individual time steps.
       - If the Euclidean-diff images show abnormal patterns, this indicates a global structural anomaly of the window; in this case, you should mark all 32 time steps as anomalous (all 1s).

2. Root Cause Analysis
   - If any time step is anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The time-series diff maps (1)-(3)
       - The Euclidean-diff maps (4)-(6)

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


# ============================================================
# Prompt：3张“时序差值图”(value/diff/var)
#       + 3张 MI 关系差图(MI diff, value/diff/var)
# ============================================================
DIRECT_PROMPT_GRAY_PLUS_MI = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 6 images describing a TEST window (32 time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST-NORMAL time-series-diff map (value)
    - A matrix image: rows = variables, columns = time steps (length 32).
    - First, the value sequences are normalized per variable to [0, 1] for the TEST window.
    - Then the value sequences are normalized per variable to [0, 1] for the matched NORMAL window.
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(2) TEST-NORMAL time-series-diff map (diff)
    - Same format as (1), but computed on first-difference sequences.
    - First, first-difference sequences are computed for the TEST window:
      diff[t] = value[t] - value[t-1].
    - Then the same first-difference sequences are computed for the matched NORMAL window.
    - Then each is normalized per variable to [0, 1].
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0,
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(3) TEST-NORMAL time-series-diff map (rolling variance)
    - Same format as (1), but computed on rolling-variance sequences.
    - First, rolling-variance sequences (window size 7) are computed for the TEST window.
    - Then the same rolling-variance sequences are computed for the matched NORMAL window.
    - Then each is normalized per variable to [0, 1].
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0,
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(4) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, a mutual-information matrix between value sequences is computed for the TEST window.
    - Then the same mutual-information matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.

(5) TEST MI-diff (diff)
    - Same as (4), but mutual information is computed on first-difference sequences(value at t − value at t−1), separately for TEST and NORMAL.

(6) TEST MI-diff (var)
    - Same as (4), but mutual information is computed on rolling-variance sequences(local volatility over time with window size 7).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the time-series-diff maps (value, diff, and rolling variance), decide for each of the 32 time steps whether it is normal (0) or anomalous (1).
   - The output should be a binary array of length 32.
   - IMPORTANT:
       - The time-series-diff maps allow you to assess anomalies at individual time steps.
       - If the MI-diff images show abnormal patterns, this indicates a global structural anomaly of the window; in this case, you should mark all 32 time steps as anomalous (all 1s).

2. Root Cause Analysis
   - If any time step is anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The time-series diff maps (1)-(3)
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

# ============================================================
# Prompt：gray-only（只看 3 张“时序差值图”）
# ============================================================
DIRECT_PROMPT_GRAY_ONLY = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 3 images describing a TEST window (32 time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST-NORMAL time-series-diff map (value)
    - A matrix image: rows = variables, columns = time steps (length 32).
    - First, the value sequences are normalized per variable to [0, 1] for the TEST window.
    - Then the value sequences are normalized per variable to [0, 1] for the matched NORMAL window.
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(2) TEST-NORMAL time-series-diff map (diff)
    - Same format as (1), but computed on first-difference sequences.
    - First, first-difference sequences are computed for the TEST window:
      diff[t] = value[t] - value[t-1].
    - Then the same first-difference sequences are computed for the matched NORMAL window.
    - Then each is normalized per variable to [0, 1].
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0,
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

(3) TEST-NORMAL time-series-diff map (rolling variance)
    - Same format as (1), but computed on rolling-variance sequences.
    - First, rolling-variance sequences (window size 7) are computed for the TEST window.
    - Then the same rolling-variance sequences are computed for the matched NORMAL window.
    - Then each is normalized per variable to [0, 1].
    - This image is the difference (TEST minus NORMAL), further mapped to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0,
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the deviation from NORMAL.

Variable names appear on the axis. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the time-series-diff maps (value, diff, and rolling variance), decide for each of the 32 time steps whether it is normal (0) or anomalous (1).
   - The output should be a binary array of length 32.
   - IMPORTANT:
       - The time-series-diff maps allow you to assess anomalies at individual time steps.

2. Root Cause Analysis
   - If any time step is anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The time-series diff maps (1)-(3)

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

DIRECT_PROMPT_EU_ONLY = """
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
DIRECT_PROMPT_MI_ONLY = """
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
DIRECT_PROMPT_EU_PLUS_MI = """
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
DIRECT_PROMPT_VALUE_ONLY = """
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
def build_direct_prompt_gray_plus_rel() -> str:
    return DIRECT_PROMPT_GRAY_PLUS_REL
# def build_direct_prompt_rel_only() -> str:
#     return DIRECT_PROMPT_REL_ONLY
def build_direct_prompt_gray_plus_eu() -> str:
    return DIRECT_PROMPT_GRAY_PLUS_EU
def build_direct_prompt_gray_plus_mi() -> str:
    return DIRECT_PROMPT_GRAY_PLUS_MI
def build_direct_prompt_gray_only() -> str:
    return DIRECT_PROMPT_GRAY_ONLY
def build_direct_prompt_eu_only() -> str:
    return DIRECT_PROMPT_EU_ONLY
def build_direct_prompt_mi_only() -> str:
    return DIRECT_PROMPT_MI_ONLY
def build_direct_prompt_eu_plus_mi() -> str:
    return DIRECT_PROMPT_EU_PLUS_MI
def build_direct_prompt_value_only() -> str:
    return DIRECT_PROMPT_VALUE_ONLY

client = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# client = OpenAI(
#     api_key="",
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

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


def call_vlm_gray_plus_rel(
    prompt: str,
    img_diff_value: Path,
    img_diff_diff: Path,
    img_diff_var: Path,
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
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_value)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_diff)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_var)}},

        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_var_test)}},

        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_var_test)}}
    ]
    print(f"the images input to LLM:")
    print(img_diff_value)
    print(img_diff_diff)
    print(img_diff_var)
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
        model="qwen3-vl-235b-a22b-thinking",
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    # completion = client.chat.completions.create(
    #     model="gemini-3-pro-preview",
    #     messages=[{"role": "user", "content": content}],
    #     temperature=0.0,
    # )
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
                "mode": "gray_plus_rel",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text


def call_vlm_gray_only(
    prompt: str,
    img_diff_value: Path,
    img_diff_diff: Path,
    img_diff_var: Path,
    save_dir: Path,
    window_id: int,
    normal_window_idx: int,
    normal_window_name: str,
    eu_folder: str,
    mi_folder: str,
):
    content = [
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_value)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_diff)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_var)}},
        {"type": "text", "text": prompt},
    ]
    print(f"the images input to LLM:")
    print(img_diff_value)
    print(img_diff_diff)
    print(img_diff_var)
    label_int = 0
    label_seq = [0] * WINDOW_SIZE
    root_causes = []
    text = ""
    label_str = "normal"

    # try:
    completion = client.chat.completions.create(
        model="qwen3-vl-235b-a22b-thinking",
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    raw = completion.choices[0].message.content
    text = "".join(x.get("text", str(x)) for x in raw) if isinstance(raw, list) else raw
    try:
        label_int, label_seq, root_causes = _parse_vlm_json(text)
        label_str = "anomaly" if label_int == 1 else "normal"
    except Exception as e:
        print(f"[WARN] JSON parse failed (gray-only) for window {window_id}: {e}")
        label_int = 0
        label_seq = [0] * WINDOW_SIZE
        root_causes = []
        label_str = "normal"
    # except Exception as e:
    #     print(f"[WARN] VLM API call failed (gray-only): {e}")
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
                "mode": "gray_only",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text


def call_vlm_gray_plus_eu(
    prompt: str,
    img_diff_value: Path,
    img_diff_diff: Path,
    img_diff_var: Path,
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
    # 只传：3张 TS + 3张 EU + prompt
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_value)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_diff)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_var)}},

        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_eu_diff_var_test)}},
    ]
    print(f"the images input to LLM:")
    print(img_diff_value)
    print(img_diff_diff)
    print(img_diff_var)
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
        model="qwen3-vl-235b-a22b-thinking",
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    raw = completion.choices[0].message.content
    text = "".join(x.get("text", str(x)) for x in raw) if isinstance(raw, list) else raw
    try:
        label_int, label_seq, root_causes = _parse_vlm_json(text)
        label_str = "anomaly" if label_int == 1 else "normal"
    except Exception as e:
        print(f"[WARN] JSON parse failed (gray+eu) for window {window_id}: {e}")
        label_int = 0
        label_seq = [0] * WINDOW_SIZE
        root_causes = []
        label_str = "normal"
    # except Exception as e:
    #     print(f"[WARN] VLM API call failed (gray+eu): {e}")
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
                "mode": "gray_plus_eu",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text


def call_vlm_gray_plus_mi(
    prompt: str,
    img_diff_value: Path,
    img_diff_diff: Path,
    img_diff_var: Path,
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
    # 只传：3张 TS + 3张 MI + prompt
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_value)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_diff)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_diff_var)}},

        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_val_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_diff_test)}},
        {"type": "image_url", "image_url": {"url": _to_b64(img_mi_diff_var_test)}},
    ]
    print(f"the images input to LLM:")
    print(img_diff_value)
    print(img_diff_diff)
    print(img_diff_var)
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
        model="qwen3-vl-235b-a22b-thinking",
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    raw = completion.choices[0].message.content
    text = "".join(x.get("text", str(x)) for x in raw) if isinstance(raw, list) else raw
    try:
        label_int, label_seq, root_causes = _parse_vlm_json(text)
        label_str = "anomaly" if label_int == 1 else "normal"
    except Exception as e:
        print(f"[WARN] JSON parse failed (gray+mi) for window {window_id}: {e}")
        label_int = 0
        label_seq = [0] * WINDOW_SIZE
        root_causes = []
        label_str = "normal"
    # except Exception as e:
    #     print(f"[WARN] VLM API call failed (gray+mi): {e}")
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
                "mode": "gray_plus_mi",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text

def call_vlm_eu_only(
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
        model="qwen3-vl-235b-a22b-thinking",
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    # completion = client.chat.completions.create(
    #     model="gemini-3-pro-preview",
    #     reasoning_effort="low",
    #     messages=[{"role": "user", "content": content}],
    #     temperature=0.0,
    # )
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
                "mode": "eu_only",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text
def call_vlm_mi_only(
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
        model="qwen3-vl-235b-a22b-thinking",
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    # completion = client.chat.completions.create(
    #     model="gemini-3-pro-preview",
    #     reasoning_effort="low",
    #     messages=[{"role": "user", "content": content}],
    #     temperature=0.0,
    # )
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
                "mode": "mi_only",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text
def call_vlm_eu_plus_mi(
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
        model="qwen3-vl-235b-a22b-thinking",
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    time_e=time.time()
    time_sec=time_e-time_s
    # completion = client.chat.completions.create(
    #     model="gemini-3-pro-preview",
    #     reasoning_effort="low",
    #     messages=[{"role": "user", "content": content}],
    #     temperature=0.0,
    # )
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
                "mode": "eu_plus_mi",
                "time":time_sec
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text
def call_vlm_value_only(
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
        model="qwen3-vl-235b-a22b-thinking",
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
    )
    # completion = client.chat.completions.create(
    #     model="gemini-3-pro-preview",
    #     reasoning_effort="low",
    #     messages=[{"role": "user", "content": content}],
    #     temperature=0.0,
    # )
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
                "mode": "value_only",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return label_int, root_causes, text