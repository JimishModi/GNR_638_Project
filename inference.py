"""
GNR638 Project 2 — Visual MCQ Solver (Score-Optimized)
Author: Jimish Modi (24B1506)

Score-optimization pipeline:
  1. High-resolution image input (max_pixels = 1024 * 28 * 28)
  2. Constrained decoding over {1,2,3,4,5} via logit restriction
     - Eliminates hallucinations (no -1 penalty possible)
     - Yields calibrated per-option probabilities
  3. Temperature scaling on logits for better calibration (T = 1.5)
  4. Expected-value skip rule: answer iff 1.25*p_best - 0.25 > 0  (p_best > 0.20)
  5. CoT fallback for mid-confidence cases (0.20 < p_best < 0.60)
     - Generates short reasoning, then re-runs constrained decoding
  6. Two-shot in-context examples in the system prompt
     - One conceptual, one math/numerical

Usage:
    python inference.py --test_dir /path/to/test/data
"""

import argparse
import os
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
OUTPUT_CSV = os.path.join(os.getcwd(), "submission.csv")

IMAGES_DIR = os.path.join(TEST_DIR, "images")
if not os.path.isdir(IMAGES_DIR):
    IMAGES_DIR = TEST_DIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model_weights")

print(f"Test dir   : {TEST_DIR}")
print(f"Images dir : {IMAGES_DIR}")
print(f"Model path : {MODEL_PATH}")
print(f"Output     : {OUTPUT_CSV}")


# ── Score-Optimization Hyperparameters ────────────────────────────────────────
# Decision rule for the negative-marking score (+1 / -0.25 / 0 / -1):
#   EV(answer) = 1.25 * p_best - 0.25  →  answer iff p_best > 0.20
ANSWER_THRESHOLD     = 0.20    # below this, we skip (output 5)
COT_LOW              = 0.20    # < this → skip outright after Stage A
COT_HIGH             = 0.60    # > this → commit after Stage A (no CoT)
TEMPERATURE          = 1.5     # softens logits for better calibration
COT_MAX_NEW_TOKENS   = 160
MAX_PIXELS           = 1024 * 28 * 28


# ── Config Patch ───────────────────────────────────────────────────────────────
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
processor = AutoProcessor.from_pretrained(PATCHED_DIR, max_pixels=MAX_PIXELS)
print("Processor loaded.")

print("Loading model (this may take a few minutes)...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cuda"
)
model.eval()
print(f"Model loaded. GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n")


# ── Token IDs for {1,2,3,4,5} (constrained decoding) ──────────────────────────
def _digit_token_id(d: str) -> int:
    """Get the single token id for a digit. Tries plain digit first."""
    ids = processor.tokenizer.encode(d, add_special_tokens=False)
    return ids[0]

ALLOWED_IDS = [_digit_token_id(str(i)) for i in range(1, 6)]
print(f"Allowed digit token ids (1..5): {ALLOWED_IDS}\n")


# ── Few-Shot Examples (text-only, embedded in instruction) ────────────────────
INSTRUCTION = (
    "You are an expert in deep learning. The image contains one multiple-choice "
    "question with four options (1, 2, 3, 4). Pick the single correct option.\n\n"
    "Examples of the kind of reasoning required:\n"
    "  Example A (conceptual): Q. Which activation suffers from the dying-neuron "
    "problem? Options: 1) Sigmoid 2) Tanh 3) ReLU 4) Softmax. Answer: 3\n"
    "  Example B (numerical): Q. Conv layer, input 32x32x3, 16 filters of size 5x5, "
    "stride 1, no padding. Output spatial size? Options: 1) 28x28 2) 32x32 "
    "3) 30x30 4) 27x27. Reasoning: (32-5)/1+1 = 28. Answer: 1\n\n"
    "Now read the question in the image and reply with ONLY the single digit "
    "(1, 2, 3, or 4) of the correct option. If you are not at all sure, reply 5."
)

COT_INSTRUCTION = (
    "You are an expert in deep learning. The image contains one multiple-choice "
    "question with four options. Reason step by step in 2-3 short sentences "
    "(do arithmetic explicitly if needed), then on a NEW final line write exactly:\n"
    "Answer: X\n"
    "where X is 1, 2, 3, or 4."
)


# ── Stage A: Constrained Decoding ─────────────────────────────────────────────
def constrained_decode(image_path: str, instruction: str):
    """One forward pass; returns (probs over {1,2,3,4,5}, best_option_int, p_best)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text": instruction}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model(**inputs)

    next_logits = out.logits[0, -1, :]                   # logits for next token
    restricted = next_logits[ALLOWED_IDS] / TEMPERATURE  # temperature scaling
    probs = torch.softmax(restricted, dim=-1).float().cpu()

    p_answers = probs[:4]                                # options 1..4
    best_idx  = int(torch.argmax(p_answers).item())
    return probs, best_idx + 1, float(p_answers[best_idx])


# ── Stage B: CoT + constrained final answer ───────────────────────────────────
def cot_then_constrained(image_path: str):
    """Generate short reasoning, then constrained-decode the final digit."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text": COT_INSTRUCTION}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    # Generate reasoning
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=COT_MAX_NEW_TOKENS,
                             do_sample=False)
    reasoning = processor.batch_decode(
        gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()

    # Append reasoning + "Answer: " and constrained-decode the digit
    forced_text = text + reasoning.rstrip() + "\nAnswer: "
    inputs2 = processor(text=[forced_text], images=image_inputs,
                        videos=video_inputs, padding=True,
                        return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs2)
    next_logits = out.logits[0, -1, :]
    restricted = next_logits[ALLOWED_IDS] / TEMPERATURE
    probs = torch.softmax(restricted, dim=-1).float().cpu()

    p_answers = probs[:4]
    best_idx  = int(torch.argmax(p_answers).item())
    return probs, best_idx + 1, float(p_answers[best_idx])


# ── Full pipeline per image ───────────────────────────────────────────────────
def answer_mcq(image_path: str) -> int:
    # Stage A: fast constrained decoding
    probs_a, opt_a, p_a = constrained_decode(image_path, INSTRUCTION)

    # High confidence → commit
    if p_a >= COT_HIGH:
        return opt_a if (1.25 * p_a - 0.25) > 0 else 5

    # Very low confidence → skip without spending CoT compute
    if p_a < COT_LOW:
        return 5

    # Mid confidence → CoT fallback, then re-decide
    probs_b, opt_b, p_b = cot_then_constrained(image_path)
    if (1.25 * p_b - 0.25) > 0:
        return opt_b
    return 5


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
