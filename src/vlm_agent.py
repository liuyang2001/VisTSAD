import json
import re
import time
from openai import OpenAI

PROMPT_ALL = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 6 images describing a TEST window ({window_size} time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the Euclidean distance between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
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
    - Same as (1), but the Euclidean distance is computed on rolling-variance sequences(local volatility over time with window size 16).

(4) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the mutual information between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
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
    - Same as (4), but the mutual information is computed on rolling-variance sequences(local volatility over time with window size 16).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the Euclidean-diff images and MI-diff images (value, diff, and rolling variance), decide for the {window_size} time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length {window_size}.

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
- Briefly describe how you used the images to decide the {window_size}-step anomaly labels.
- Briefly describe how you used the images to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {{
  "Label": [l_1, l_2, ..., l_{window_size}],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}}

Rules:
- "Label" MUST be an array of exactly {window_size} integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""

PROMPT_EU_ONLY = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 3 images describing a TEST window ({window_size} time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the Euclidean distance between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
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
    - Same as (1), but the Euclidean distance is computed on rolling-variance sequences(local volatility over time with window size 16).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the Euclidean-diff images (value, diff, and rolling variance), decide for the {window_size} time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length {window_size}.

2. Root Cause Analysis
   - If all time steps are anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The Euclidean-diff maps (1)-(3)

=====================================================
REASONING FORMAT (REQUIRED)
=====================================================

Analysis Process:
- Briefly describe how you used the images to decide the {window_size}-step anomaly labels.
- Briefly describe how you used the images to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {{
  "Label": [l_1, l_2, ..., l_{window_size}],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}}

Rules:
- "Label" MUST be an array of exactly {window_size} integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""

PROMPT_MI_ONLY = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 3 images describing a TEST window ({window_size} time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the mutual information between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
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
    - Same as (1), but the mutual information is computed on rolling-variance sequences(local volatility over time with window size 16).

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the MI-diff images (value, diff, and rolling variance), decide for the {window_size} time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length {window_size}.

2. Root Cause Analysis
   - If all time steps are anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from:
       - The MI-diff maps (1)-(3)

=====================================================
REASONING FORMAT (REQUIRED)
=====================================================

Analysis Process:
- Briefly describe how you used the images to decide the {window_size}-step anomaly labels.
- Briefly describe how you used the images to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {{
  "Label": [l_1, l_2, ..., l_{window_size}],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}}

Rules:
- "Label" MUST be an array of exactly {window_size} integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""

PROMPT_VALUE_ONLY = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 2 images describing a TEST window ({window_size} time steps) and its matched NORMAL reference window.

=====================================================
IMAGES (in order)
=====================================================

(1) TEST Euclidean-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the Euclidean distance between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
    - Then the same pairwise Euclidean-distance matrix is computed for the NORMAL window.
    - This image is the difference (TEST minus NORMAL), further normalized to the range [-1, 1].
    - The color scale is:
        - White  ≈ 0  (almost no difference),
        - Blue   ≈ -1,
        - Red    ≈  1.
      The closer to white, the smaller the difference; the closer to blue or red, the larger the difference from NORMAL.

(2) TEST MI-diff (value)
    - A square matrix (variables × variables).
    - First, for the TEST window, compute the mutual information between every pair of variables based on their {window_size}-step value sequences, forming an N×N matrix (N = number of variables).
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
   - Based on the Euclidean-diff image and MI-diff image, decide for the {window_size} time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length {window_size}.

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
- Briefly describe how you used the images to decide the {window_size}-step anomaly labels.
- Briefly describe how you used the images to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {{
  "Label": [l_1, l_2, ..., l_{window_size}],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}}

Rules:
- "Label" MUST be an array of exactly {window_size} integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""

PROMPT_NO_RESIDUAL = """
You are an expert in anomaly detection and root cause analysis for multivariate time-series, using only images.

You will be given 6 images describing a TEST window ({window_size} time steps).

=====================================================
IMAGES (in order)
=====================================================

(1) TEST Euclidean (value)
    - A square matrix (variables × variables).
    - Computed for the TEST window: the Euclidean distance between every pair of variables based on their {window_size}-step value sequences.
    - Normalized to the range [0, 1] and mapped to Grayscale (0-255).
    - The color scale is:
        - Black  ≈ 0  (Small distance / Low value),
        - White  ≈ 1  (Large distance / High value).
    - This image represents the raw structural relationships within the test window itself.

(2) TEST Euclidean (diff)
    - Same as (1), but computed on first-difference sequences(value at t − value at t−1).

(3) TEST Euclidean (var)
    - Same as (1), but computed on rolling-variance sequences(local volatility over time with window size 16).

(4) TEST MI (value)
    - A square matrix (variables × variables).
    - Computed for the TEST window: the mutual information between every pair of variables based on their {window_size}-step value sequences.
    - Normalized to the range [0, 1] and mapped to Grayscale.
    - The color scale is:
        - Black  ≈ 0  (Low correlation),
        - White  ≈ 1  (High correlation).

(5) TEST MI (diff)
    - Same as (4), but computed on first-difference sequences.

(6) TEST MI (var)
    - Same as (4), but computed on rolling-variance sequences.

Variable names appear on axes. Use them exactly.

=====================================================
YOUR TASK
=====================================================

For each TEST window:

1. Time-step anomaly labeling
   - Based on the Euclidean and MI images (value, diff, and rolling variance), decide for the {window_size} time steps whether they are normal (0) or anomalous (1).
   - The output should be a binary array of length {window_size}.

2. Root Cause Analysis
   - If all time steps are anomalous, identify the most likely root-cause variables.
   - If all time steps are normal, output [] for "root_cause_variables".
   - Use information from the 6 maps.

=====================================================
REASONING FORMAT (REQUIRED)
=====================================================

Analysis Process:
- Briefly describe how you used the images to decide the {window_size}-step anomaly labels.
- Briefly describe how you used the images to decide the root-cause variables.

=====================================================
FINAL OUTPUT — STRICT JSON FORMAT
=====================================================

Final Answer: {{
  "Label": [l_1, l_2, ..., l_{window_size}],
  "root_cause_variables": ["var_name_1", "var_name_2", ...]
}}

Rules:
- "Label" MUST be an array of exactly {window_size} integers (0 or 1).
- "root_cause_variables" MUST be a JSON array. If all labels are 0, output [].
- Do NOT output anything after this JSON block.
"""


class VLMAgent:
    def __init__(self, config, window_size=64):

        self.client = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
        self.model_name = config.get('model_name', "gemini-3-pro-preview")
        self.window_size = window_size 
        
        self.prompt_map = {
            "all": PROMPT_ALL,
            "eu_only": PROMPT_EU_ONLY,
            "mi_only": PROMPT_MI_ONLY,
            "value_only": PROMPT_VALUE_ONLY,
            "no_residual": PROMPT_NO_RESIDUAL
        }
        
        self.mode_keys = {
            "all": ["eu_val", "eu_diff", "eu_var", "mi_val", "mi_diff", "mi_var"],
            "eu_only": ["eu_val", "eu_diff", "eu_var"],
            "mi_only": ["mi_val", "mi_diff", "mi_var"],
            "value_only": ["eu_val", "mi_val"],
            "no_residual": ["eu_val", "eu_diff", "eu_var", "mi_val", "mi_diff", "mi_var"]
        }

    def _parse_json(self, text):
        match = re.search(r"Final Answer:\s*(\{.*\})", text, re.DOTALL)
        if not match:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
        
        if match:
            try:
                data = json.loads(match.group(1))
                if "Label" not in data: data["Label"] = [0] * self.window_size
                if "root_cause_variables" not in data: data["root_cause_variables"] = []
                
                labels = data["Label"]
                if not isinstance(labels, list): labels = [0] * self.window_size
                
                if len(labels) < self.window_size: 
                    labels = labels + [0] * (self.window_size - len(labels))
                elif len(labels) > self.window_size: 
                    labels = labels[:self.window_size]
                
                data["Label"] = labels
                
                return data
            except json.JSONDecodeError:
                pass
        
        return {"Label": [0]*self.window_size, "root_cause_variables": [], "error": "parse_failed"}

    def analyze(self, mode, images_map):
        if mode not in self.prompt_map:
            raise ValueError(f"Unknown mode: {mode}")

        prompt_template = self.prompt_map[mode]
        prompt_text = prompt_template.format(window_size=self.window_size)
        
        required_keys = self.mode_keys[mode]
        content = [{"type": "text", "text": prompt_text}]
        
        valid_imgs = 0
        for k in required_keys:
            if k in images_map:
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": images_map[k]}
                })
                valid_imgs += 1
        
        if valid_imgs == 0:
            return {"Label": [0]*self.window_size, "root_cause_variables": [], "latency": 0.0}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                start_time = time.perf_counter()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0
                )
                
                end_time = time.perf_counter()
                elapsed = round(end_time - start_time, 4)
                
                raw_text = response.choices[0].message.content
                
                result = self._parse_json(raw_text)
                
                result["latency"] = elapsed
                result["raw_model_response"] = raw_text 
                
                return result
            
            except Exception as e:
                print(f"  [VLM Error] Attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(2)
        
        return {"Label": [0]*self.window_size, "root_cause_variables": [], "error": "api_failed", "latency": 0.0}