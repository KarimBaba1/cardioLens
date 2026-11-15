# CardioLens

An explainable 10-year coronary heart disease (CHD) risk screening assistant, comparing
machine learning and a large language model against the clinical Framingham Risk Score.
Academic prototype (UFCE3P-30-3, AY26/27) — **not for clinical use**.

## Quick start (web app)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_app.sh            # then open http://127.0.0.1:8000
```

The plain-language summary uses a **local** LLM through [Ollama](https://ollama.com)
(`qwen2.5:7b`, ~4.7 GB; `qwen2.5:3b` for slower laptops via `CARDIOLENS_LLM=qwen2.5:3b`).
Install Ollama, then `ollama pull qwen2.5:7b`. Without Ollama the app still works and shows
a template summary.

Demo shortcuts: `/?example=0|1|2` loads a dataset patient (lower / average / higher risk),
add `&summary=1` to also write the summary; `/#evidence` opens the model comparison.

## Results (shared held-out test patients, n = 294, 46 CHD events)

| Approach | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** (used in app) | 0.219 | 0.348 | 0.269 | **0.690** |
| Random Forest | 0.245 | 0.261 | 0.253 | 0.676 |
| Framingham Risk Score (BMI) | 0.229 | 0.543 | 0.323 | 0.684 |
| LLM zero-shot (Qwen2.5-7B, 4-bit) | 0.187 | 0.435 | 0.261 | 0.591 |
| LLM fine-tuned (QLoRA) | 0.217 | 0.435 | 0.290 | 0.680 |

Thresholds were chosen on the validation set. Full tables: `results/comparison/`.

## Repository

| Path | What |
|---|---|
| `data/raw/framingham.csv` | Kaggle Framingham dataset (4,238 rows) |
| `src/prepare_data.py` | Checks + stratified 70/15/15 split (seed 42) → `data/processed/` |
| `src/preprocessing.py` | Shared features, train-only imputation/scaling, label-free patient text |
| `src/logistic_regression.py`, `src/random_forest.py` | ML models |
| `src/framingham_score.py` | Clinical baseline (D'Agostino 2008, BMI model) |
| `src/llm_zero_shot.py`, `src/llm_finetune.py` | LLM approaches (run on a Colab T4 GPU) |
| `src/evaluate.py` | Comparison + patient-group metrics → `results/comparison/` |
| `src/train_final_model.py` | Saves the app model → `models/` |
| `app/` | FastAPI backend + single-page UI |

## Limitations

Moderate accuracy (ROC-AUC ≈ 0.69); historical US cohort; the Framingham BMI score predicts
general cardiovascular disease, not CHD only; LLM results use a 300-patient test subset and a
4-bit model; factor influences and what-if results are associations, not causal effects;
group results reveal uneven performance but do not prove fairness.
