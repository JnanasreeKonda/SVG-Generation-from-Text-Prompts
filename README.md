# SVG-Generation-from-Text-Prompts

# SVG Generation from Text Prompts
### QLoRA Fine-Tuning of Qwen2.5-1.5B-Instruct · Kaggle Competition

---

## Table of Contents

1. [Overview](#overview)
2. [Competition Constraints](#competition-constraints)
3. [Approach Summary](#approach-summary)
4. [Repository Structure](#repository-structure)
5. [Requirements](#requirements)
6. [Dataset Setup](#dataset-setup)
7. [How to Run](#how-to-run)
8. [Notebook Walkthrough (Cell by Cell)](#notebook-walkthrough)
9. [Configuration Reference](#configuration-reference)
10. [Model Architecture & Training Details](#model-architecture--training-details)
11. [Inference Pipeline](#inference-pipeline)
12. [Results](#results)
13. [Ablation Studies](#ablation-studies)
14. [Known Issues & Tips](#known-issues--tips)
15. [Citation](#citation)

---

## Overview

This notebook is a complete, reproducible solution for the Kaggle competition
**"SVG Generation from Text Prompts"**, in which participants fine-tune a small
language model (≤ 4B parameters) to generate valid SVG code from natural language
descriptions.

**Task:** Given a text prompt (e.g., *"A simple smiley face with a wide open mouth
and straight eyes"*), generate valid SVG XML code that visually matches the description.

**Scoring:** Submissions are evaluated on a composite metric combining:
- Visual similarity to reference SVGs (SSIM + Edge F1)
- Structural quality of the generated SVG
- Compactness bonus for shorter, cleaner SVGs

**Our approach:** 4-bit QLoRA fine-tuning of `Qwen/Qwen2.5-1.5B-Instruct` on a
cleaned 12,000-sample subset of the provided 50,000 training pairs.

---

## Competition Constraints

All generated SVGs **must** satisfy the following or receive a score of **0**:

| Constraint | Value |
|---|---|
| Must be valid XML | Starting with `<svg` |
| Maximum characters | 8,000 |
| Maximum path elements | 256 |
| Canvas size | 256×256 pixels (`viewBox="0 0 256 256"`) |
| Allowed tags | `svg`, `g`, `path`, `rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon`, `defs`, `use`, `symbol`, `clipPath`, `mask`, `linearGradient`, `radialGradient`, `stop`, `text`, `tspan`, `title`, `desc`, `style`, `pattern`, `marker`, `filter` |
| External data | Not allowed |
| Model size | ≤ 4B parameters (≤ 2B recommended) |

---

## Approach Summary

| Component | Choice | Reason |
|---|---|---|
| **Base model** | `Qwen/Qwen2.5-1.5B-Instruct` | Strong instruction-following at 1.54B params; large vocabulary efficiently encodes SVG syntax |
| **Fine-tuning** | QLoRA (4-bit NF4, r=32, α=64) | Fits in GPU memory; 2.34% trainable params; built-in regularization prevents overfitting |
| **Training data** | 12,000 cleaned samples (from 50,000) | Compute-optimal; diminishing returns beyond 12k |
| **Decoding** | Greedy (`do_sample=False`) | Lowest fallback rate; structured output prefers consistency over diversity |
| **Repair pipeline** | Two-pass retry + XML repair | Recovers near-valid outputs; 98.8% final validity rate |

---

## Repository Structure

```
.
├── SVG_Generation_from_Text_Prompts.ipynb   # Main notebook (all-in-one)
├── README.md                                # This file
├── svg_qwen15b_instruct_ckpt/               # Training checkpoints (auto-created)
├── svg_qwen15b_instruct_adapter/            # Saved LoRA adapter weights (auto-created)
└── submission.csv                           # Final output (auto-created)
```

> **Note:** `train.csv`, `test.csv`, and `sample_submission.csv` must be placed in
> the working directory before running. On Kaggle, these are automatically available
> at `/kaggle/input/`.

---

## Requirements

### Hardware

| Component | Minimum | Used in Development |
|---|---|---|
| GPU | NVIDIA T4 (16 GB) | NVIDIA H100 80GB HBM3 |
| RAM | 16 GB | — |
| Disk | 20 GB free | — |

> On a T4 you may need to reduce `BATCH_SIZE` to 1 and `NUM_TRAIN_SAMPLES` to 8,000
> to avoid OOM errors. On an A100 or H100, the default settings run comfortably.

### Software

Install all dependencies by running **Cell 0** in the notebook, or manually:

```bash
pip install -U transformers peft accelerate bitsandbytes datasets \
               scikit-learn lxml cairosvg
```

| Package | Purpose |
|---|---|
| `transformers` | Model loading, tokenization, `Trainer` |
| `peft` | LoRA adapter via `LoraConfig`, `get_peft_model` |
| `accelerate` | Multi-device / mixed-precision training |
| `bitsandbytes` | 4-bit NF4 quantization (`BitsAndBytesConfig`) |
| `datasets` | HuggingFace `Dataset` wrapper for training data |
| `scikit-learn` | Stratified train/validation split |
| `lxml` | SVG XML parsing and validation |
| `cairosvg` | SVG rendering (optional, for visual inspection) |

> **Python version:** 3.11 (Kaggle default). The notebook is tested on Kaggle's
> standard GPU kernel environment.

---

## Dataset Setup

The competition provides three CSV files. Place them in the same directory as the
notebook (they are automatically available on Kaggle):

### `train.csv` — 50,000 rows

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique UUID row identifier |
| `prompt` | string | Natural language description (~20 words average) |
| `svg` | string | Ground-truth SVG code |

### `test.csv` — 1,000 rows

| Column | Type | Description |
|---|---|---|
| `id` | string | Unique UUID row identifier |
| `prompt` | string | Natural language description (no SVG provided) |

### `sample_submission.csv`

Template CSV with `id` and `svg` columns. Your `submission.csv` must match this format.

---

## How to Run

### On Kaggle (recommended)

1. Fork or import the notebook into a Kaggle notebook session.
2. Enable **GPU accelerator** (T4, P100, or A100).
3. Attach the competition dataset so that `train.csv` and `test.csv` are accessible.
4. Click **Run All** — the notebook runs end-to-end in approximately 115 minutes on H100.
5. Download `submission.csv` from the output files panel.

### On Google Colab

1. Upload the notebook to Colab.
2. Set runtime to **GPU** (A100 recommended via Colab Pro).
3. Upload `train.csv` and `test.csv` to the Colab file system or mount Google Drive.
4. Run all cells in order.
5. The last cell downloads `submission.csv` automatically.

### Locally

```bash
# 1. Clone / download the notebook
# 2. Install dependencies
pip install -U transformers peft accelerate bitsandbytes datasets \
               scikit-learn lxml cairosvg

# 3. Place train.csv and test.csv in the working directory

# 4. Convert notebook to script (optional)
jupyter nbconvert --to script SVG_Generation_from_Text_Prompts.ipynb

# 5. Run
python SVG_Generation_from_Text_Prompts.py
```

---

## Notebook Walkthrough

The notebook is organized into 12 numbered cells. Here is what each one does:

### Cell 0 — Install Dependencies
Installs all required Python packages via `pip`. Safe to re-run.

```python
!pip -q install -U transformers peft accelerate bitsandbytes datasets \
                   scikit-learn lxml cairosvg
```

---

### Cell 1 — Configuration
**Single source of truth for all hyperparameters.** Edit this cell to run experiments
or ablations. Key settings:

```python
MODEL_ID           = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_TRAIN_SAMPLES  = 12000      # samples drawn from train.csv
VAL_FRAC           = 0.05       # 5% held out for validation
SEED               = 42

# SVG filtering
MIN_SVG_CHARS      = 60         # drop degenerate SVGs
MAX_TRAIN_SVG_CHARS= 5200       # cap training SVG length
MAX_SUBMIT_SVG_CHARS= 8000      # competition hard limit

# Training
MAX_SEQ_LENGTH     = 1024
BATCH_SIZE         = 2
GRAD_ACCUM         = 8          # effective batch = 16
NUM_EPOCHS         = 2
LR                 = 1e-4

# Inference
GEN_MAX_NEW_TOKENS = 650        # first-pass generation budget
GEN_MAX_NEW_TOKENS_RETRY = 900  # retry budget for failed rows
INFERENCE_BATCH    = 8
```

---

### Cell 2 — Imports, Seeds, and Device Detection
Sets all random seeds for reproducibility and detects the available GPU.

```
device: cuda
NVIDIA H100 80GB HBM3, 81559 MiB, 4 MiB used
```

---

### Cell 3 — SVG Utility Functions
Defines the helper functions used throughout the pipeline:

| Function | Purpose |
|---|---|
| `count_path_elements(svg)` | Counts `<path>` elements via regex |
| `normalize_svg(svg)` | Injects `xmlns`, `viewBox`, strips control chars, compacts whitespace |
| `validate_submit_svg(svg)` | Checks all competition constraints; returns `(bool, reason)` |
| `repair_for_submit(raw)` | Extracts SVG from raw model output; appends `</svg>` if missing |

---

### Cell 4 — Load and Inspect Data
Loads `train.csv` and `test.csv`, prints shapes and sample rows.

```
train shape: (45731, 3)    # after dropping malformed rows
test shape:  (1000, 2)
train cols:  ['id', 'prompt', 'svg']
test cols:   ['id', 'prompt']
```

> The nominal 50,000 rows reduce to ~45,731 after the initial null/empty drop.

---

### Cell 5 — Data Cleaning Pipeline
Applies four cleaning steps to produce the `svg_clean` column:

1. **Parse validation** — drops rows that fail `xml.etree.ElementTree` parsing
2. **Length filter** — drops SVGs shorter than `MIN_SVG_CHARS` (60 chars)
3. **Path element filter** — drops SVGs with more than 256 `<path>` elements
4. **Normalization** — runs `normalize_svg()` to standardize `xmlns`, `viewBox`, whitespace
5. **Length cap** — truncates to `MAX_TRAIN_SVG_CHARS` (5,200 chars)
6. **Stratified sampling** — samples `NUM_TRAIN_SAMPLES` rows, stratified by SVG length quartile

---

### Cell 6 — Tokenizer Setup and Prompt Template
Loads the Qwen2.5 tokenizer and defines the instruction format used for training
and inference:

```
System:    Generate only valid compact SVG. Output only SVG XML.
           Start with <svg and end with </svg>.
User:      <prompt text>
Assistant: <svg_clean>          ← included during training, generated during inference
```

The Qwen2.5 chat template wraps this in `<|im_start|>` / `<|im_end|>` tokens automatically.

---

### Cell 7 — Build HuggingFace Datasets
Splits the cleaned data into train/validation sets and tokenizes:

```
train rows: 11,400
val rows:      600
token length:  1,024 (truncated + padded)
```

Labels are set equal to `input_ids` (full causal LM loss over the entire sequence,
including the prompt portion).

---

### Cell 8 — Load Model with QLoRA
Loads `Qwen2.5-1.5B-Instruct` in 4-bit NF4 quantization and attaches the LoRA adapter:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
```

```
trainable params: 36,929,536 / 1,580,643,840 (2.34%)
```

---

### Cell 9 — Training
Runs the HuggingFace `Trainer` for 2 epochs (~70 minutes on H100). Training and
validation loss are logged every 250 steps:

| Step | Train Loss | Val Loss |
|---|---|---|
| 250  | 0.4430 | 0.4364 |
| 500  | 0.4007 | 0.4006 |
| 750  | 0.3576 | 0.3825 |
| 1000 | 0.3384 | 0.3722 |
| 1250 | 0.3634 | 0.3677 |
| 1426 | 0.3812 | **0.3670** |

The adapter is saved to `./svg_qwen15b_instruct_adapter/` on completion.

---

### Cell 10 — Inference Helpers (Sanity Check)
Defines `generate_batch()` and runs it on 5 validation samples to verify the
pipeline is working before full inference.

Key settings:
- `model.eval()` + `torch.inference_mode()` — disables gradient tracking
- `tokenizer.padding_side = "left"` — required for correct batched decoding
- `do_sample=False` — greedy decoding for consistency

---

### Cell 11 — Full Inference on Test Set
Runs batched inference over all 1,000 test prompts with a two-pass retry:

1. **Pass 1:** Generate with `GEN_MAX_NEW_TOKENS=650`, batch size 8
2. **Validate:** Each output checked against all competition constraints
3. **Pass 2 (retry):** Failed rows re-generated with `GEN_MAX_NEW_TOKENS_RETRY=900`, batch size 4
4. **Fallback:** Rows still invalid receive a minimal white-rectangle SVG

```
Inference complete.
Rows: 1000  |  Unique SVGs: 436  |  Fallback count: 12  |  Median SVG length: 355
```

---

### Cell 12 — Save and Download submission.csv
Writes the final `submission.csv` (columns: `id`, `svg`) and triggers a browser
download on Colab/Kaggle.

---

## Configuration Reference

All parameters are defined in **Cell 1**. Common modifications:

### Reduce memory usage (e.g., for T4 GPU)
```python
BATCH_SIZE         = 1       # was 2
NUM_TRAIN_SAMPLES  = 8000    # was 12000
MAX_SEQ_LENGTH     = 768     # was 1024
INFERENCE_BATCH    = 4       # was 8
```

### Increase training data
```python
NUM_TRAIN_SAMPLES  = 20000
NUM_EPOCHS         = 3
```

### Try a different model
```python
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"   # larger, better quality
# or
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" # smaller, faster
```

### Enable sampling at inference
```python
# In Cell 10 / generate_batch():
do_sample    = True
temperature  = 0.7
top_p        = 0.95
```

---

## Model Architecture & Training Details

### Base Model: Qwen2.5-1.5B-Instruct

| Property | Value |
|---|---|
| Architecture | Transformer decoder (Qwen2.5) |
| Parameters | 1,540,000,000 (~1.54B) |
| Vocabulary size | 151,643 tokens |
| Context window | 32,768 tokens (we use 1,024) |
| Quantization | 4-bit NF4 + double quantization |
| Compute dtype | float16 |

### LoRA Adapter

| Property | Value |
|---|---|
| Rank `r` | 32 |
| Alpha `α` | 64 (effective LR scale = α/r = 2.0) |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 36,929,536 (2.34% of total) |

### Optimizer & Schedule

| Property | Value |
|---|---|
| Optimizer | paged_adamw_8bit |
| Learning rate | 1e-4 |
| LR scheduler | cosine |
| Warmup steps | 80 |
| Effective batch size | 16 (batch=2 × grad_accum=8) |
| Max gradient norm | 0.3 |
| Mixed precision | fp16 |
| Gradient checkpointing | Enabled |

---

## Inference Pipeline

```
Test prompt
    │
    ▼
build_prompt_text()          ← Apply Qwen chat template (system + user turn)
    │
    ▼
tokenizer()                  ← Left-pad to max_length=192; return_tensors="pt"
    │
    ▼
model.generate()             ← Greedy, max_new_tokens=650, use_cache=True
    │
    ▼
extract_svg()                ← Regex: find <svg...>...</svg> or partial <svg...
    │
    ▼
safe_xml_repair()            ← Strip bad chars, inject xmlns/viewBox, close tag
    │
    ▼
validate_submit_svg()        ← Check length, path count, XML validity
    │
   / \
PASS   FAIL
  │      │
  │    retry with max_new_tokens=900
  │      │
  │    validate again
  │     / \
  │   PASS FAIL
  │     │    │
  │     │  FALLBACK_SVG (white rectangle)
  │     │    │
  └─────┴────┘
    submission.csv
```

---

## Results

| Metric | Value |
|---|---|
| Final training loss | 0.3812 |
| Final validation loss | 0.3670 |
| Test rows processed | 1,000 / 1,000 |
| Valid SVGs (all constraints passed) | **988 / 1,000 (98.8%)** |
| Fallback SVG rows | 12 / 1,000 (1.2%) |
| Unique SVGs generated | 436 / 1,000 (43.6%) |
| Median SVG length | 355 characters |
| Max SVG length | 1,329 characters |
| Total training time | ~70 minutes (H100) |
| Total inference time | ~44 minutes (H100) |

---


*GPU used: NVIDIA H100 80GB HBM3 · Framework: HuggingFace Transformers + PEFT + BitsAndBytes*
