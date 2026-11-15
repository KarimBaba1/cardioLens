"""Fine-tuned LLM risk prediction (QLoRA; runs on a Colab T4 GPU).

Same model, same prompt and same scoring rule as src/llm_zero_shot.py, so the
only difference between the two runs is the fine-tuning.

Training signal: for each TRAINING patient, a 2-way cross-entropy between the
model's next-token logits for "Yes" and "No" (exactly what is read out at
inference) and the true TenYearCHD label. Positives are up-weighted by
neg/pos (~5.6) because only ~15% of patients are positive.

Data integrity:
  * trains on data/processed/train.csv only
  * val.csv is used to choose the best epoch (val ROC-AUC) and the threshold
  * the 300-patient test subset is scored once, at the end
  * no test labels or test rows are used in any training step

Expected layout on the Colab VM (uploaded by the driver):
    /content/cardiolens/src/{preprocessing,llm_zero_shot,llm_finetune}.py
    /content/cardiolens/data/processed/{train,val,llm_test_subset}.csv
Outputs: /content/cardiolens/results/llm_finetuned/
"""

import random
import sys
import time

sys.path.insert(0, "/content/cardiolens/src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from sklearn.metrics import (f1_score, precision_score, recall_score,  # noqa: E402
                             roc_auc_score)

# Importing llm_zero_shot installs the libraries and gives us the SAME prompt,
# model loader, scoring function and threshold rule used for the zero-shot run.
from llm_zero_shot import (MODEL_ID, QUESTION, ROOT, SYSTEM,  # noqa: E402
                           best_f1_threshold, load_model, score)
from preprocessing import TARGET, row_to_text  # noqa: E402

OUT = ROOT / "results" / "llm_finetuned"
SEED = 42
EPOCHS = 3
BATCH = 8
ACCUM = 2
LR = 1e-4
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM")


def build_prompts(df, tok):
    return [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": QUESTION.format(record=row_to_text(r))}],
        tokenize=False, add_generation_prompt=True)
        for _, r in df.iterrows()]


def lora_state(model):
    return {k: v.detach().cpu().clone()
            for k, v in model.named_parameters() if "lora_" in k}


def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    val = pd.read_csv(ROOT / "data/processed/val.csv").sort_values("patient_id")
    test = pd.read_csv(ROOT / "data/processed/llm_test_subset.csv"
                       ).sort_values("patient_id")
    print(f"train={len(train)} val={len(val)} test_subset={len(test)}", flush=True)

    tok, model = load_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**LORA))
    model.print_trainable_parameters()

    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    prompts = build_prompts(train, tok)
    # class index 0 = Yes (CHD), 1 = No
    labels = torch.tensor((1 - train[TARGET].to_numpy()).astype(int))
    pos_w = float((train[TARGET] == 0).sum() / (train[TARGET] == 1).sum())
    weight = torch.tensor([pos_w, 1.0]).cuda()
    print(f"positive class weight = {pos_w:.2f}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    steps_total = EPOCHS * (len(prompts) // (BATCH * ACCUM))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps_total))
    scaler = torch.amp.GradScaler("cuda")

    best = {"auc": -1.0, "epoch": 0, "state": None, "val_p": None}
    history = []
    t0 = time.time()
    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = np.random.permutation(len(prompts))
        running, n_run = 0.0, 0
        for b, i in enumerate(range(0, len(order) - BATCH + 1, BATCH)):
            idx = order[i:i + BATCH]
            enc = tok([prompts[j] for j in idx], return_tensors="pt",
                      padding=True).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                last = model(**enc, logits_to_keep=1).logits[:, -1, :]
            two = torch.stack([last[:, yes_id], last[:, no_id]], 1).float()
            loss = F.cross_entropy(two, labels[idx].cuda(), weight=weight)
            scaler.scale(loss / ACCUM).backward()
            running += loss.item(); n_run += 1
            if (b + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step(); step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{steps_total} "
                          f"loss {running / n_run:.4f} "
                          f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
                    running, n_run = 0.0, 0
        model.eval()
        val_p = score(val, tok, model)
        auc = roc_auc_score(val[TARGET], val_p)
        history.append({"epoch": epoch, "val_roc_auc": auc})
        print(f"== epoch {epoch}: val ROC-AUC {auc:.4f} "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        if auc > best["auc"]:
            best.update(auc=auc, epoch=epoch, state=lora_state(model), val_p=val_p)

    # restore the best epoch (chosen on validation only), then score the test subset
    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in best["state"]:
                v.copy_(best["state"][k].to(v.device, v.dtype))
    model.eval()
    print(f"best epoch = {best['epoch']} (val ROC-AUC {best['auc']:.4f})", flush=True)
    test_p = score(test, tok, model)

    val_p = best["val_p"]
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
    pd.DataFrame(history).to_csv(OUT / "epoch_history.csv", index=False)
    torch.save(best["state"], OUT / "lora_adapter.pt")
    y = test[TARGET]
    metrics = pd.DataFrame([{
        "model": f"{MODEL_ID} fine-tuned QLoRA (epoch {best['epoch']})",
        "threshold": threshold,
        "val_roc_auc": best["auc"],
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
    print(f"total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
