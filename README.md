# GNR638 Project 2 — Visual MCQ Solver

**Author:** Jimish Modi (Roll: 24B1506)
**Course:** GNR638 — Deep Learning and Visual Computing, IIT Bombay (2025–26)
**Task:** Solve deep-learning MCQs presented as PNG images under a negative-marking scoring scheme.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Scoring Scheme & EV Analysis](#scoring-scheme--ev-analysis)
3. [Model Choice](#model-choice)
4. [Architecture & Pipeline](#architecture--pipeline)
5. [Design Decisions Explained](#design-decisions-explained)
6. [Accuracy Results](#accuracy-results)
7. [The No-Dataset Problem](#the-no-dataset-problem)
8. [Challenges & Solutions](#challenges--solutions)
9. [Repository Structure](#repository-structure)
10. [Setup & Usage](#setup--usage)

---

## Problem Statement

Given a folder of PNG images, each containing a deep-learning multiple-choice question with 4 options, predict the correct option (1, 2, 3, or 4). Output 5 to skip if unsure.

**Scoring (negative marking):**

| Prediction | Score |
|---|---|
| Correct | +1 |
| Wrong | −0.25 |
| Skip (5) | 0 |
| Hallucinated (not in 1–5) | −1 |

> ⚠️ No dataset is provided. Participants must create or source their own data.

**Hardware:** NVIDIA L40s, 48GB VRAM, CUDA 12.6
**Time limit:** 1 hour per run, ≤50 questions

---

## Scoring Scheme & EV Analysis

The scoring rule is not symmetric — wrong answers cost only 0.25 but hallucinations cost 1.0. The key insight is framing answering vs. skipping as an **Expected Value (EV)** decision:

Let `p_best` = model's probability for its top choice among options 1–4.

```
EV(answer) = (+1) × p_best + (−0.25) × (1 − p_best)
           = 1.25 × p_best − 0.25

EV(skip)   = 0

Answer iff EV(answer) > EV(skip):
    1.25 × p_best − 0.25 > 0
    p_best > 0.20
```

This gives the **mathematically optimal threshold of p_best > 0.20** — derived directly from the scoring rule, requiring no empirical tuning.

---

## Model Choice

**[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)** (Alibaba, 7B parameters)

Chosen for:
- **Native vision-language capability** — reads image and text simultaneously in one forward pass
- **Strong document/OCR understanding** — MCQs rendered as images require robust text extraction
- **Fits in 48GB VRAM** without quantization — full precision inference on L40s
- **Open-source weights** — downloadable via HuggingFace CLI, no API calls at inference time
- **High-resolution input** — supports up to `1024 × 28 × 28` pixels per image

---

## Architecture & Pipeline

The solver uses a **two-stage cascade** per question:

```
Input Image
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  STAGE A: Fast Constrained Decoding                  │
│                                                      │
│  1. Build prompt: image + two-shot instruction       │
│  2. Single forward pass (no token generation)        │
│  3. Extract next-token logits                        │
│  4. Restrict to token IDs for {1, 2, 3, 4, 5}       │
│  5. Apply temperature scaling (T = 1.5)              │
│  6. Softmax → calibrated probability distribution    │
│  7. p_best = max(p_1, p_2, p_3, p_4)                │
└───────────────────────┬──────────────────────────────┘
                        │
           ┌────────────┼────────────┐
           │            │            │
      p_best        p_best       p_best
      ≥ 0.60       [0.20, 0.60)  < 0.20
           │            │            │
           ▼            ▼            ▼
       Commit     ┌──────────┐    Skip (5)
       answer     │ STAGE B  │
                  │   CoT    │
                  │ Fallback │
                  │          │
                  │ Generate │
                  │ reasoning│
                  │ (≤160 tok│
                  │          │
                  │ Re-run   │
                  │constrained
                  │ decoding │
                  │          │
                  │ Re-apply │
                  │ EV rule  │
                  └────┬─────┘
                       │
                Answer or Skip (5)
```

### Stage A: Constrained Decoding

Instead of free-form generation, a **single forward pass** is run and logits are extracted only for the 5 relevant tokens `{1, 2, 3, 4, 5}`:

```python
next_logits = out.logits[0, -1, :]                   # logits for the next token
restricted  = next_logits[ALLOWED_IDS] / TEMPERATURE  # restrict + temperature scale
probs       = torch.softmax(restricted, dim=-1)        # probabilities over {1..5}
```

This **eliminates hallucinations entirely** — the model cannot output anything outside {1, 2, 3, 4, 5}, making the −1 penalty architecturally impossible.

### Stage B: Chain-of-Thought (CoT) Fallback

For mid-confidence cases (0.20 ≤ p_best < 0.60):
1. Ask the model to reason step-by-step in 2–3 sentences with explicit arithmetic
2. Append the generated reasoning to the prompt context
3. Re-run constrained decoding on the extended context
4. Re-apply the EV threshold on the updated probability

This allows the model to "think out loud" before committing, recovering accuracy on questions where initial confidence was ambiguous.

### Two-Shot In-Context Examples

Two examples are embedded in the system prompt to prime both reasoning styles:

- **Example A (conceptual):** *"Which activation suffers from the dying-neuron problem?"* → ReLU (option 3)
- **Example B (numerical):** *"Conv layer, input 32×32×3, 16 filters 5×5, stride 1, no padding. Output size?"* → (32-5)/1+1=28 → 28×28×16 (option 1)

---

## Design Decisions Explained

| Decision | Rationale |
|---|---|
| Constrained decoding over free generation | Eliminates −1 hallucination penalty entirely; 10× faster than autoregressive generation |
| Single forward pass for Stage A | No token generation loop needed; dramatically faster per image |
| Temperature T = 1.5 | Counteracts known LLM overconfidence; better calibration for EV decision rule |
| EV threshold p > 0.20 | Mathematically derived from +1/−0.25 rule; provably optimal |
| CoT only for mid-confidence (0.20–0.60) | Saves compute on easy questions; adds reasoning only where needed |
| `device_map="cuda"` | Single L40s (48GB); entire 7B model fits without splitting |
| `max_pixels = 1024×28×28` | Maximum resolution for reading fine-grained text in MCQ images |
| `torch_dtype="auto"` | Lets PyTorch auto-select bf16/fp16 for the L40s |
| Preprocessor config patch | Fixes `image_processor_type` mismatch between model weights and some `transformers` versions |

---

## Accuracy Results

### Standard Test Set (20 questions: conceptual + simple math)

| Metric | Value |
|---|---|
| Correct | 19 / 20 |
| Wrong | 1 |
| Skipped | 0 |
| **Hallucinated** | **0** |
| Accuracy (attempted) | 95.0% |
| **Final Score** | **18.75 / 20** |

### Hard Test Set (20 questions: multi-step math + adversarial intuition)

Designed to stress-test the pipeline — questions where the "obvious" answer is a distractor.

| Metric | Value |
|---|---|
| Correct | 17 / 20 |
| Wrong | 3 |
| Skipped | 0 |
| **Hallucinated** | **0** |
| Accuracy (attempted) | 85.0% |
| **Final Score** | **16.25 / 20** |

**Key observation:** Hallucinations = 0 across both test sets. The 3 errors on the hard set were multi-step arithmetic questions — not hallucinations — confirming the constrained decoding guarantee held perfectly.

---

## The No-Dataset Problem

The project provided **zero labeled data** — no training set, no validation set, only a sample directory structure. This creates a fundamental challenge: how do you validate your pipeline and tune hyperparameters without ground truth?

### Strategy 1: Synthetic Data Generation

A custom script (`generate_test_data.py`) renders MCQ images programmatically using `matplotlib`, exactly matching the TA's image format:

```python
# Renders question + 4 options as a PNG image
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
ax.text(0.05, 0.92, question_text, fontsize=14, weight="bold")
# ... render options at decreasing y positions
plt.savefig(save_path, facecolor="white")
```

This produced:
- 20 deep-learning MCQs (conceptual + numerical) rendered as PNG
- `test.csv` (image names, no answers)
- `ground_truth.csv` (for local evaluation only)

### Strategy 2: Mathematical Hyperparameter Derivation

Instead of searching over hyperparameter values (which requires data), the skip threshold was **derived from first principles**:

```
Score rule: +1 / −0.25 / 0 / −1
EV(answer) = 1.25 × p_best − 0.25
EV(skip)   = 0
Optimal threshold: p_best > 0.20
```

No labeled data required. The threshold is provably optimal for the given scoring scheme.

### Strategy 3: Conservative CoT Boundary

The `COT_HIGH = 0.60` threshold was set conservatively based on reasoning: questions where a 7B VLM is >60% confident on deep-learning MCQs are almost certainly correct. The synthetic test confirmed this intuition.

### Strategy 4: Stress Testing with Adversarial Questions

A second harder test set (`generate_hard_test.py`) was created with:
- Multi-step math (dilated conv geometry, LSTM parameter counts, attention FLOPs)
- Adversarial intuition questions (ResNet degradation, BatchNorm train/test behavior, optimizer generalization)

This validated that the pipeline degrades gracefully on harder questions (85% vs 95%) without ever hallucinating.

---

## Challenges & Solutions

### 1. Out-of-Memory on Single T4 GPU
**Problem:** Loading Qwen2.5-VL-7B with `device_map="cuda"` on a Kaggle T4 (16GB VRAM) caused OOM.
**Solution:** Used `device_map="auto"` for Kaggle testing (model splits across 2×T4). Production grading uses L40s (48GB), so `device_map="cuda"` is correct and used in the final submission.

### 2. Preprocessor Config Mismatch
**Problem:** Downloaded model weights listed `image_processor_type: Qwen2_5_VLImageProcessor` which some `transformers` versions don't recognize.
**Solution:** Config-patch step copies all JSON configs to `/tmp/` and corrects the processor type:
```python
if config.get("image_processor_type") != "Qwen2VLImageProcessor":
    config["image_processor_type"] = "Qwen2VLImageProcessor"
```

### 3. No Internet During Inference
**Problem:** Grading machine has no internet when `inference.py` runs — model cannot be downloaded on-the-fly.
**Solution:** `setup.bash` downloads all ~14GB of weights via `huggingface-cli` during setup. `inference.py` reads weights entirely from `./model_weights/` — zero network calls at inference time.

### 4. Hallucination Risk from Free-Form Generation
**Problem:** Unconstrained VLM generation can produce *"The answer is option three"*, *"C"*, or off-topic text — all penalized at −1.
**Solution:** Constrained decoding restricts output to token IDs for `{1, 2, 3, 4, 5}` at the logit level. No regex, no post-processing, no edge cases — hallucination is architecturally impossible.

### 5. Missing `torchvision` Dependency
**Problem:** `qwen_vl_utils` internally imports `torchvision` for image preprocessing. Omitting it from `setup.bash` caused an import error on the grading machine.
**Fix:** Add `torchvision` explicitly alongside `torch` in the pip install list.

### 6. `argparse` Fails Inside Jupyter Notebook
**Problem:** Pasting `inference.py` code into a Kaggle cell and running it causes `argparse` to parse the notebook kernel's `sys.argv` instead of `--test_dir`.
**Solution:** Always run as a shell command: `!python inference.py --test_dir ./path`

---

## Repository Structure

```
GNR_638_Project/
├── inference.py        ← Main solver: reads images → runs model → writes submission.csv
├── setup.bash          ← Environment setup: clone repo, create conda env, install deps, download weights
├── requirements.txt    ← Python dependencies list
└── README.md           ← This file
```

---

## Setup & Usage

`setup.bash` handles the full environment setup automatically:

```bash
# Step 1: Clone repository
git clone https://github.com/JimishModi/GNR_638_Project /tmp/gnr638_repo
cp /tmp/gnr638_repo/inference.py .

# Step 2: Create conda environment (Python 3.11)
conda create -n gnr_project_env python=3.11 -y

# Step 3: Install all dependencies
conda run -n gnr_project_env pip install \
    torch torchvision \
    git+https://github.com/huggingface/transformers \
    accelerate qwen-vl-utils pillow pandas huggingface_hub

# Step 4: Download Qwen2.5-VL-7B-Instruct weights (~14GB)
conda run -n gnr_project_env huggingface-cli download \
    Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./model_weights
```

Run inference:

```bash
conda activate gnr_project_env
python inference.py --test_dir /absolute/path/to/test/data
# Output: submission.csv written to current working directory
```

### Expected input structure:
```
test_dir/
├── test.csv          # Required column: image_name
└── images/
    ├── image_1.png
    ├── image_2.png
    └── ...
```

### Output:
```
submission.csv        # Columns: id, image_name, option
```

---

## Hardware Requirements

| Component | Specification |
|---|---|
| GPU | NVIDIA L40s (48GB VRAM) |
| CUDA | 12.x (12.6 recommended) |
| System RAM | 16GB+ |
| Disk | ~15GB free (model weights) |
| Python | 3.11 |
| Conda | Any recent version |

---

## References

- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [Qwen2.5-VL HuggingFace Model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [qwen_vl_utils](https://github.com/QwenLM/Qwen2.5-VL)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
