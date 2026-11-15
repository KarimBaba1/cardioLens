# CardioLens — presenter guide (10–15 min)

All numbers below were produced by the app itself.

## Before the presentation

**Day before**
- [ ] `git pull` on the presenting laptop, then `./run_app.sh` once — check the summary says
      *"Checked: numbers match the model"* (not *Template summary*).
- [ ] Plug-in charger ready: on battery the CPU is slower and summaries take longer.

**10 minutes before**
- [ ] Close heavy apps (browser tabs, IDE) — the LLM needs ~5 GB of memory.
- [ ] Run `./run_app.sh`, open http://127.0.0.1:8000
- [ ] Wait for the green dot **"qwen2.5:7b ready (local)"** in the summary card (≈25 s).
- [ ] Open three tabs so you never type URLs live:
  1. `http://127.0.0.1:8000/` — blank form for the live entry
  2. `http://127.0.0.1:8000/?example=2` — higher-risk patient, already assessed
  3. `http://127.0.0.1:8000/#evidence` — model comparison
- [ ] Browser zoom 110–125% so the room can read it.

## Running order

| Time | Section | Show |
|---|---|---|
| 0:00–2:00 | Problem & data | slides |
| 2:00–5:00 | Results: who wins? | slides, then **tab 3** |
| 5:00–11:00 | Live demo | **tab 1**, then **tab 2** |
| 11:00–13:00 | Limitations & ethics | **About & limits** tab or slides |
| 13:00–15:00 | Conclusion, questions | |

## Results (tab 3, *Model evidence*)

- "Every approach was scored on the **same 294 test patients** none of them saw during training."
- Point at ROC-AUC: Logistic Regression **0.690**, Framingham **0.684**, fine-tuned LLM **0.680**,
  Random Forest **0.676**, zero-shot LLM **0.591**.
- Message: *the fine-tuned LLM only caught up after ~100 minutes of GPU training; the zero-shot
  LLM is clearly weaker; a simple model and the clinical formula do as well.* So LLMs are not
  the right **predictor** here — but they are useful as an **explainer** (you'll see that next).
- Scroll to **patient groups**: recall for women is **0.21** vs **0.47** for men; under-45s have only
  15 events (*limited data*). "Overall scores hide uneven performance."

## Live demo

**A. Validation (tab 1, 30 s)** — type age **75**, select *Assess risk*:
> "Age must be between 32 and 70 years, the range the model was trained on."

"The app refuses inputs the model never learned from." Change age to **55**.

**B. Live patient (tab 1, 2 min)** — enter:

| Field | Value | Field | Value |
|---|---|---|---|
| Sex | Male | Systolic BP | 150 |
| Age | 55 | Diastolic BP | 95 |
| Current smoker | Yes | Total cholesterol | 250 |
| Cigarettes/day | 20 | BMI | 28 |
| BP medication | No | Heart rate | 75 |
| Hypertension | Yes | Glucose | 85 |
| Stroke / Diabetes | No / No | Education | 2 |

Result: **32%** — "about 2× the study average of 15%, above the 20% screening threshold".
Factors chart: **blood pressure, age, smoking, sex** raise the estimate.

**C. What-if (same patient, 1 min)**
- Set *Current smoker* to **No** → **32.4% → 23.9%** (−8.5 points).
- Then lower systolic BP to **120** → **18.3%**, below the threshold.
- Say: *"This is how the model responds — not proof that quitting or treatment has this effect."*

**D. Summary (1 min)** — select **Write summary** (~15 s; talk while it writes):
"A local Qwen model runs on this laptop — no patient data leaves the computer. The app
checks the text: if it changes a number or gives treatment advice, it shows a safe template."

**E. Higher-risk example (tab 2, 30 s)** — **40%**, driven by blood pressure (195 mmHg) and age (63).

## Likely questions

| Question | Answer |
|---|---|
| Why not use the best LLM? | It only matched logistic regression after 100 min of GPU training, can't explain itself exactly, and is far slower. |
| Is 0.69 ROC-AUC good? | Moderate. Similar to the clinical Framingham formula on this data, but many future cases are missed — hence a screening aid, not a diagnosis. |
| Did you leak test data? | No. Test labels never appeared in training, tuning, or prompts; thresholds were chosen on a separate validation set. |
| Why is the LLM test only 300 patients? | GPU time on the free tier; all approaches are compared on the same shared patients. |
| Framingham predicts CHD? | The BMI version predicts general cardiovascular disease — broader than our target; we state this as a limitation. |
| Would this work in Hong Kong? | Unknown: historical US cohort. It would need validation on local data. |
| Why does smoking show as one factor? | The model splits smoking across two correlated inputs; we combine them so the explanation is not misleading. |

## If something goes wrong

| Problem | Fix |
|---|---|
| Summary shows *Template summary* | Keep going — it is a safe, correct summary. Mention the fallback is a deliberate safety feature. |
| Summary is slow (>30 s) | Talk through the factor chart meanwhile; on battery, plug in. |
| App won't start | `./run_app.sh` in a terminal and read the message; port busy → close the old terminal. |
| Laptop fails entirely | Use screenshots / screen recording made the day before (recommended backup). |
