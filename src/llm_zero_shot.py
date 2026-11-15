"""Zero-shot LLM risk prediction (runs on a Colab T4 GPU, not locally).

Model: Qwen2.5-7B-Instruct, loaded in 4-bit so it fits the 15 GB T4.
The model gets ONE patient description (src/preprocessing.row_to_text, which
never contains the label) and must answer Yes or No. The risk score is
P(Yes) / (P(Yes) + P(No)) read from the next-token probabilities, so
ROC-AUC is a real probability-based metric.

Data integrity:
  * no examples, no labels and no test information in any prompt
  * the classification threshold is chosen on the VALIDATION set only
  * predictions are made for the fixed 300-patient test subset

Expected layout on the Colab VM (uploaded by the driver):
    /content/cardiolens/src/preprocessing.py
    /content/cardiolens/data/processed/{val,llm_test_subset}.csv
Outputs: /content/cardiolens/results/llm_zero_shot/
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes", "accelerate", "transformers"], check=True)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (f1_score, precision_score, recall_score,  # noqa: E402
                             roc_auc_score)
from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          BitsAndBytesConfig)

ROOT = Path("/content/cardiolens")
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import TARGET, row_to_text  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUT = ROOT / "results" / "llm_zero_shot"
BATCH = 8

SYSTEM = ("You are a careful clinical risk-screening assistant. You estimate "
          "whether a person is likely to have a coronary heart disease (CHD) "
          "event within the next 10 years, using only the information given.")
QUESTION = ("Patient record:\n{record}\n\n"
            "Will this patient have a coronary heart disease event within "
            "10 years? Answer with exactly one word: Yes or No.")


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16),
        device_map="auto", dtype=torch.float16)
    model.eval()
    return tok, model


@torch.no_grad()
def score(df: pd.DataFrame, tok, model) -> np.ndarray:
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": QUESTION.format(record=row_to_text(r))}],
        tokenize=False, add_generation_prompt=True)
        for _, r in df.iterrows()]
    out = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt",
                  padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        two = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=1)
        out.extend(torch.softmax(two, dim=1)[:, 0].cpu().numpy())
        if (i // BATCH) % 10 == 0:
            print(f"  scored {min(i + BATCH, len(prompts))}/{len(prompts)}",
                  flush=True)
    return np.array(out)


def best_f1_threshold(y, p) -> float:
    cands = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 91)))
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def main() -> None:
    val = pd.read_csv(ROOT / "data/processed/val.csv").sort_values("patient_id")
    test = pd.read_csv(ROOT / "data/processed/llm_test_subset.csv"
                       ).sort_values("patient_id")
    print(f"val={len(val)} test_subset={len(test)}", flush=True)

    tok, model = load_model()
    print("model loaded; GPU mem GB:",
          round(torch.cuda.max_memory_allocated() / 1e9, 1), flush=True)

    print("scoring validation...", flush=True)
    val_p = score(val, tok, model)
    print("scoring test subset...", flush=True)
    test_p = score(test, tok, model)

    threshold = best_f1_threshold(val[TARGET], val_p)
    pred = (test_p >= threshold).astype(int)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": val["patient_id"].to_numpy(),
                  "actual": val[TARGET].to_numpy(),
                  "probability": val_p}).to_csv(OUT / "val_scores.csv", index=False)
    pd.DataFrame({"patient_id": test["patient_id"].to_numpy(),
                  "actual": test[TARGET].to_numpy(),
                  "predicted": pred, "probability": test_p}
                 ).to_csv(OUT / "predictions.csv", index=False)
    y = test[TARGET]
    metrics = pd.DataFrame([{
        "model": f"{MODEL_ID} zero-shot (4-bit)",
        "threshold": threshold,
        "val_roc_auc": roc_auc_score(val[TARGET], val_p),
        "accuracy": float((pred == y).mean()),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_score": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, test_p),
    }])
    metrics.to_csv(OUT / "metrics.csv", index=False)
    print(metrics.drop(columns="model").round(3).to_string(index=False))
    print("score summary (test): min %.3f median %.3f max %.3f" %
          (test_p.min(), np.median(test_p), test_p.max()))


if __name__ == "__main__":
    main()
