import os, json, re, torch, glob
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"

ADAPTER_DIR = "/workspace/insight_cxr_3b/adapter/"
MODEL_ID    = "Qwen/Qwen2.5-VL-3B-Instruct"
TEST_JSONL  = "/workspace/insight_cxr/processed/test.jsonl"
OUTPUT_DIR  = "/workspace/insight_cxr_3b/evaluation/full_test/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are an expert radiologist AI assistant. "
    "Given a chest X-ray image, provide accurate clinical findings. "
    "When referring to a specific region, output its bounding box "
    "immediately after using the format: <box>[x1, y1, x2, y2]</box> "
    "where coordinates are absolute pixels on a 448x448 image. "
    "Be factual. Do not hallucinate findings not visible in the image."
)

# ── Load model ────────────────────────────────────────────────
print("Loading 3B model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
base_model.config.use_cache = True
ft_model  = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
ft_model.eval()
processor = AutoProcessor.from_pretrained(
    ADAPTER_DIR,
    min_pixels=256*28*28,
    max_pixels=1280*28*28,
    padding_side="left",
)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
print("Model ready.")

# ── Load all test records ─────────────────────────────────────
with open(TEST_JSONL) as f:
    all_test = [json.loads(l) for l in f if l.strip()]
print(f"Total test records: {len(all_test):,}")

# ── Resume from checkpoints ───────────────────────────────────
checkpoint_files = sorted(glob.glob(OUTPUT_DIR + "results_checkpoint_*.csv"))
already_done = set()
if checkpoint_files:
    done_dfs = [pd.read_csv(f) for f in checkpoint_files]
    done_df  = pd.concat(done_dfs, ignore_index=True).drop_duplicates(subset=["index"])
    already_done = set(done_df["index"].tolist())
    print(f"Resuming — already processed: {len(already_done):,} records")
    print(f"Remaining: {len(all_test) - len(already_done):,} records")
else:
    print("Starting fresh evaluation...")

# ── Helper functions ──────────────────────────────────────────
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def run_inference(image_path, question):
    try:
        img = Image.open(image_path).convert("RGB")
    except:
        return ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt], images=[img],
        return_tensors="pt", padding=True
    ).to("cuda")
    with torch.no_grad():
        out = ft_model.generate(
            **inputs, max_new_tokens=300,
            do_sample=False, repetition_penalty=1.1,
        )
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

# ── Evaluation loop ───────────────────────────────────────────
print(f"\nEvaluating remaining records...")
print("Progress saved every 500 samples.\n")

records    = []
iou_scores = []
bbox_found = 0
SAVE_EVERY = 500
new_count  = 0

for i, sample in enumerate(tqdm(all_test)):
    # Skip already processed
    if i in already_done:
        continue

    response = run_inference(sample["image_path"], sample["question"])
    if not response:
        continue

    clean      = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    answer     = re.sub(r'<box>\[.*?\]</box>', '', clean).strip()
    pred_boxes = re.findall(
        r'<box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>', response
    )

    iou = 0.0
    if pred_boxes and sample["bboxes"]:
        bbox_found += 1
        pred = [int(x) for x in pred_boxes[0]]
        iou  = compute_iou(pred, sample["bboxes"][0])
    iou_scores.append(iou)

    records.append({
        "index"      : i,
        "image_path" : sample["image_path"],
        "question"   : sample["question"],
        "reference"  : sample["answer"],
        "prediction" : answer,
        "gt_bboxes"  : json.dumps(sample["bboxes"]),
        "pred_bboxes": json.dumps(pred_boxes),
        "bbox_found" : int(len(pred_boxes) > 0),
        "iou"        : float(iou),
        "iou_50"     : int(iou >= 0.50),
        "iou_75"     : int(iou >= 0.75),
        "iou_90"     : int(iou >= 0.90),
    })

    new_count += 1
    if new_count % SAVE_EVERY == 0:
        df_temp = pd.DataFrame(records)
        ckpt_name = f"results_checkpoint_{i+1}.csv"
        df_temp.to_csv(OUTPUT_DIR + ckpt_name, index=False)
        curr_iou = sum(iou_scores)/len(iou_scores)
        print(f"\n[{i+1}/{len(all_test)}] mean IoU: {curr_iou:.3f} | bbox_found: {bbox_found}/{new_count}")

# Save any remaining new records
if records:
    df_temp = pd.DataFrame(records)
    df_temp.to_csv(OUTPUT_DIR + f"results_checkpoint_{len(all_test)}_final.csv", index=False)

# ── Combine ALL results (old + new) ──────────────────────────
print("\nCombining all results...")
all_files = sorted(glob.glob(OUTPUT_DIR + "results_checkpoint_*.csv"))
all_df    = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
all_df    = all_df.drop_duplicates(subset=["index"]).sort_values("index")
print(f"Total records combined: {len(all_df):,}")

predictions = all_df["prediction"].fillna("").tolist()
references  = all_df["reference"].fillna("").tolist()
iou_all     = all_df["iou"].tolist()
bbox_all    = all_df["bbox_found"].tolist()

# ── Final metrics ─────────────────────────────────────────────
smooth    = SmoothingFunction().method1
tok_preds = [p.lower().split() for p in predictions]
tok_refs  = [[r.lower().split()] for r in references]

bleu1 = corpus_bleu(tok_refs, tok_preds,
                    weights=(1,0,0,0), smoothing_function=smooth)
bleu4 = corpus_bleu(tok_refs, tok_preds,
                    weights=(.25,.25,.25,.25), smoothing_function=smooth)

scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = sum(scorer.score(r, p)["rougeL"].fmeasure
              for r, p in zip(references, predictions)) / len(predictions)

mean_iou   = float(np.mean(iou_all))
median_iou = float(np.median(iou_all))
iou50_rate = sum(1 for x in iou_all if x >= 0.50) / len(iou_all)
iou75_rate = sum(1 for x in iou_all if x >= 0.75) / len(iou_all)
iou90_rate = sum(1 for x in iou_all if x >= 0.90) / len(iou_all)
bbox_rate  = sum(bbox_all) / len(bbox_all)

print(f"\n{'='*60}")
print(f"FULL TEST SET EVALUATION  (n={len(all_df):,})")
print(f"{'='*60}")
print(f"BLEU-1        : {bleu1:.4f}")
print(f"BLEU-4        : {bleu4:.4f}")
print(f"ROUGE-L       : {rouge_l:.4f}")
print(f"Mean IoU      : {mean_iou:.4f}")
print(f"Median IoU    : {median_iou:.4f}")
print(f"IoU >= 0.50   : {iou50_rate*100:.2f}%")
print(f"IoU >= 0.75   : {iou75_rate*100:.2f}%")
print(f"IoU >= 0.90   : {iou90_rate*100:.2f}%")
print(f"BBox Found    : {bbox_rate*100:.2f}%")
print(f"{'='*60}")

# ── Save final results ────────────────────────────────────────
all_df.to_csv(OUTPUT_DIR + "full_test_results.csv", index=False)

metrics = {
    "total_samples" : len(all_df),
    "BLEU-1"        : round(bleu1, 4),
    "BLEU-4"        : round(bleu4, 4),
    "ROUGE-L"       : round(rouge_l, 4),
    "Mean_IoU"      : round(mean_iou, 4),
    "Median_IoU"    : round(median_iou, 4),
    "IoU_0.50"      : round(iou50_rate, 4),
    "IoU_0.75"      : round(iou75_rate, 4),
    "IoU_0.90"      : round(iou90_rate, 4),
    "BBox_Found_Pct": round(bbox_rate*100, 2),
}
with open(OUTPUT_DIR + "full_test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nResults saved → {OUTPUT_DIR}")
print("full_test_results.csv")
print("full_test_metrics.json")