"""
GNR638 Project 2 — Visual MCQ Solver
Author: Jimish Modi (24B1506)

Uses Qwen2.5-VL-7B-Instruct to answer deep learning MCQ images.
Model weights must be present in ./model_weights/ (downloaded by setup.bash).

Usage:
    python inference.py --test_dir /path/to/test/data
"""

import argparse
import os
import re
import json
import shutil
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ── Argument Parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GNR638 Project 2 — Visual MCQ Solver")
parser.add_argument("--test_dir", type=str, required=True,
                    help="Path to test directory containing test.csv and images/")
args = parser.parse_args()

TEST_DIR   = args.test_dir
TEST_CSV   = os.path.join(TEST_DIR, "test.csv")
OUTPUT_CSV = os.path.join(os.getcwd(), "submission.csv")   # saved in current dir

# Images are in test_dir/images/ — fallback to test_dir/ if images/ doesn't exist
IMAGES_DIR = os.path.join(TEST_DIR, "images")
if not os.path.isdir(IMAGES_DIR):
    IMAGES_DIR = TEST_DIR

# Model weights downloaded by setup.bash into ./model_weights/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model_weights")

print(f"Test dir   : {TEST_DIR}")
print(f"Images dir : {IMAGES_DIR}")
print(f"Model path : {MODEL_PATH}")
print(f"Output     : {OUTPUT_CSV}")


# ── Config Patch ───────────────────────────────────────────────────────────────
# Qwen2.5-VL HuggingFace weights may have wrong image_processor_type in config.
# We patch it before loading the processor.
PATCHED_DIR = "/tmp/gnr638_patched_config"
os.makedirs(PATCHED_DIR, exist_ok=True)

for fname in os.listdir(MODEL_PATH):
    if fname.endswith(".json"):
        shutil.copy(os.path.join(MODEL_PATH, fname),
                    os.path.join(PATCHED_DIR, fname))

config_path = os.path.join(PATCHED_DIR, "preprocessor_config.json")
with open(config_path) as f:
    config = json.load(f)

if config.get("image_processor_type") != "Qwen2VLImageProcessor":
    config["image_processor_type"] = "Qwen2VLImageProcessor"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("preprocessor_config.json patched.")


# ── Load Processor & Model ────────────────────────────────────────────────────
print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(
    PATCHED_DIR,
    max_pixels=256 * 28 * 28   # limits image tokens to fit in GPU memory
)
print("Processor loaded.")

print("Loading model (this may take a few minutes)...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cuda"
)
model.eval()
print(f"Model loaded. GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n")


# ── Inference Function ─────────────────────────────────────────────────────────
def answer_mcq(image_path: str) -> int:
    """
    Given a path to an MCQ image, returns predicted answer as int 1-4.
    Returns 5 (skip) if model is uncertain or output is unparseable.
    Scoring: +1 correct, -0.25 wrong, -1 hallucinated, 0 for skip(5).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": (
                    "This image contains a multiple choice question about deep learning. "
                    "Read the question and all four options carefully. "
                    "Reply with ONLY a single digit: 1, 2, 3, or 4 corresponding to the correct option. "
                    "If you are not confident, reply with 5. "
                    "Do not explain. Do not write anything else."
                )}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False
        )

    generated = output_ids[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    match = re.search(r"[1-5]", raw)
    if match:
        return int(match.group())
    return 5   # unparseable output → skip (no penalty)


# ── Main Loop ──────────────────────────────────────────────────────────────────
df = pd.read_csv(TEST_CSV)
results = []

for _, row in df.iterrows():
    image_name = row["image_name"]
    image_path = os.path.join(IMAGES_DIR, f"{image_name}.png")

    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path} → skipping (5)")
        pred = 5
    else:
        pred = answer_mcq(image_path)

    print(f"{image_name} → {pred}")
    results.append({
        "id":         image_name,
        "image_name": image_name,
        "option":     pred
    })

# ── Save Submission ────────────────────────────────────────────────────────────
submission = pd.DataFrame(results)
submission.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(submission)} predictions to {OUTPUT_CSV}")
print(submission)
