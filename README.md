# GNR638 Project 2 — Setup & Run Instructions

**Author:** Jimish Modi (24B1506)  
**Course:** GNR638, IIT Bombay

## Overview

Visual MCQ Solver using **Qwen2.5-VL-7B-Instruct** (Vision-Language Model).  
The model reads PNG images of deep learning MCQs and predicts the correct answer (1/2/3/4) or outputs 5 to skip.

---

## Setup

Run the setup script (internet required):

```bash
bash setup.bash
conda activate gnr_project_env
```

This will automatically:
1. Clone this repo and place `inference.py` in the current directory
2. Create conda environment `gnr_project_env` (Python 3.11)
3. Install all dependencies
4. Download Qwen2.5-VL-7B-Instruct weights (~14GB) into `./model_weights/`

---

## Running Inference

```bash
python inference.py --test_dir /absolute/path/to/test/data
```

**Test directory must follow this structure:**
```
test_data/
├── images/
│   ├── image_1.png
│   └── image_2.png
├── test.csv
└── sample_submission.csv
```

**Output:** `submission.csv` saved in the current working directory.

---

## Dependencies

| Package | Version |
|---|---|
| Python | 3.11 |
| transformers | latest (from git) |
| accelerate | latest |
| qwen-vl-utils | latest |
| pillow | latest |
| pandas | latest |
| huggingface_hub | latest |

---

## References

- [Qwen2.5-VL Paper](https://arxiv.org/abs/2502.13923)
- [Qwen2.5-VL HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
