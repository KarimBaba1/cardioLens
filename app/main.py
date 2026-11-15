"""CardioLens web app (FastAPI backend).

Run from the repo root:
    .venv/bin/uvicorn app.main:app --port 8000
then open http://127.0.0.1:8000

The risk estimate comes from the saved Logistic Regression model
(models/, created by src/train_final_model.py). The plain-language summary is
written by a LOCAL LLM through Ollama (default qwen2.5:7b); if Ollama is not
running or the text fails the safety checks, a fixed template is used instead.
"""

import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import FEATURES, load_split  # noqa: E402

MODEL = joblib.load(ROOT / "models" / "logistic_regression.joblib")
CARD = json.loads((ROOT / "models" / "model_card.json").read_text())
THRESHOLD = CARD["threshold"]
BASE_RATE = CARD["base_rate"]
RANGES = CARD["supported_ranges"]
TYPICAL = CARD.get("typical_ranges", {})
COEF = MODEL.named_steps["classifier"].coef_[0]
INTERCEPT = float(MODEL.named_steps["classifier"].intercept_[0])
FEATURE_ORDER = CARD["feature_order"]
TRAIN_MEAN_T = np.array(CARD["train_mean_transformed"])

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("CARDIOLENS_LLM", "qwen2.5:7b")

# ---------------------------------------------------------------- form schema
FIELDS = [
    # name, label, group, kind, required, unit, help
    ("male", "Sex", "Patient", "binary", True, "", ["Female", "Male"]),
    ("age", "Age", "Patient", "number", True, "years", None),
    ("education", "Education level", "Patient", "select", False, "",
     ["1", "2", "3", "4"]),
    ("currentSmoker", "Current smoker", "Lifestyle", "binary", True, "", ["No", "Yes"]),
    ("cigsPerDay", "Cigarettes per day", "Lifestyle", "number", True, "per day", None),
    ("BPMeds", "On blood-pressure medication", "Medical history", "binary", True, "", ["No", "Yes"]),
    ("prevalentHyp", "Diagnosed hypertension", "Medical history", "binary", True, "", ["No", "Yes"]),
    ("prevalentStroke", "Previous stroke", "Medical history", "binary", True, "", ["No", "Yes"]),
    ("diabetes", "Diabetes", "Medical history", "binary", True, "", ["No", "Yes"]),
    ("sysBP", "Systolic blood pressure", "Measurements", "number", True, "mmHg", None),
    ("diaBP", "Diastolic blood pressure", "Measurements", "number", True, "mmHg", None),
    ("totChol", "Total cholesterol", "Measurements", "number", True, "mg/dL", None),
    ("BMI", "Body-mass index", "Measurements", "number", True, "kg/m²", None),
    ("heartRate", "Resting heart rate", "Measurements", "number", False, "bpm", None),
    ("glucose", "Glucose", "Measurements", "number", False, "mg/dL", None),
]
FIELD = {f[0]: f for f in FIELDS}
LABELS = {f[0]: f[1] for f in FIELDS}

FACTOR_GROUPS = {
    "Age": ["age"],
    "Sex": ["male"],
    "Smoking": ["currentSmoker", "cigsPerDay"],
    "Blood pressure": ["sysBP", "diaBP", "prevalentHyp", "BPMeds"],
    "Cholesterol": ["totChol"],
    "Body-mass index": ["BMI"],
    "Glucose & diabetes": ["glucose", "diabetes"],
    "Previous stroke": ["prevalentStroke"],
    "Heart rate": ["heartRate"],
    "Education": ["education"],
}

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "framingham_score": "Framingham Risk Score (BMI)",
    "llm_zero_shot": "LLM zero-shot (Qwen2.5-7B)",
    "llm_finetuned": "LLM fine-tuned (Qwen2.5-7B QLoRA)",
}

app = FastAPI(title="CardioLens")


# ---------------------------------------------------------------- validation
def validate(raw: dict) -> tuple[dict, dict, list]:
    """Return (clean values, field errors, warnings)."""
    clean, errors, warnings = {}, {}, []
    for name, label, _, kind, required, unit, _opts in FIELDS:
        v = raw.get(name)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            if required:
                errors[name] = f"{label} is required."
            else:
                clean[name] = None
            continue
        try:
            num = float(v)
        except (TypeError, ValueError):
            errors[name] = f"{label} must be a number."
            continue
        if math.isnan(num) or math.isinf(num):
            errors[name] = f"{label} must be a number."
            continue
        if kind == "binary" and num not in (0, 1):
            errors[name] = f"{label} must be Yes or No."
            continue
        if name in RANGES:
            lo, hi = RANGES[name]
            if not lo <= num <= hi:
                u = f" {unit}" if unit else ""
                errors[name] = (f"{label} must be between {lo:g} and {hi:g}{u}, "
                                "the range the model was trained on.")
                continue
        clean[name] = num
    if not errors:
        if clean["diaBP"] >= clean["sysBP"]:
            errors["diaBP"] = "Diastolic pressure must be lower than systolic pressure."
        if clean["currentSmoker"] == 0 and clean["cigsPerDay"] > 0:
            errors["cigsPerDay"] = "Set cigarettes to 0, or mark the patient as a current smoker."
        if clean["currentSmoker"] == 1 and clean["cigsPerDay"] == 0:
            warnings.append("Marked as a smoker with 0 cigarettes per day; please check.")
        unusual = [LABELS[n] for n, (lo, hi) in TYPICAL.items()
                   if clean.get(n) is not None and not lo <= clean[n] <= hi]
        if unusual:
            warnings.append(f"Unusual value for {', '.join(unusual).lower()}: very few patients "
                            "like this in the training data, so the estimate is less reliable.")
        for opt in ("education", "heartRate", "glucose"):
            if clean.get(opt) is None:
                warnings.append(f"{LABELS[opt]} not given: a typical value from the "
                                "training data was used.")
    return clean, errors, warnings


# ---------------------------------------------------------------- prediction
def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def predict(clean: dict) -> dict:
    row = pd.DataFrame([{f: (np.nan if clean.get(f) is None else clean[f])
                         for f in FEATURES}])
    risk = float(MODEL.predict_proba(row[FEATURES])[0, 1])
    x = MODEL.named_steps["preprocessing"].transform(row[FEATURES])[0]
    # contribution of each factor in log-odds, relative to the average
    # training patient (exact for logistic regression)
    contrib = dict(zip(FEATURE_ORDER, COEF * (x - TRAIN_MEAN_T)))
    avg_risk = sigmoid(INTERCEPT + float(COEF @ TRAIN_MEAN_T))
    # Related inputs are summed into one clinical factor: e.g. the model splits
    # smoking across "current smoker" and "cigarettes per day" (correlated), so
    # showing them separately could wrongly suggest smoking lowers risk.
    factors = []
    for name, feats in FACTOR_GROUPS.items():
        parts = [{"label": LABELS[f], "contribution": float(contrib[f])} for f in feats]
        net = float(sum(p["contribution"] for p in parts))
        # parts that pull in opposite directions and cancel (e.g. low systolic BP
        # vs BP medication) are flagged so the chart does not hide them
        offset = len(parts) > 1 and abs(net) < 0.05 and max(abs(p["contribution"]) for p in parts) >= 0.1
        factors.append({"factor": name, "inputs": feats, "contribution": net,
                        "parts": parts, "offset": offset})
    factors.sort(key=lambda d: -max(abs(d["contribution"]),
                                    max(abs(p["contribution"]) for p in d["parts"]) if d["offset"] else 0))
    detail = [{"feature": n, "label": LABELS[n], "value": clean.get(n),
               "contribution": float(c)} for n, c in contrib.items()]
    return {
        "risk": risk,
        "threshold": THRESHOLD,
        "flagged": risk >= THRESHOLD,
        "base_rate": BASE_RATE,
        "average_patient_risk": avg_risk,
        "model": CARD["model"],
        "factors": factors,
        "detail": detail,
    }


# ---------------------------------------------------------------- LLM summary
SYSTEM_PROMPT = (
    "You write short plain-language summaries of a heart-disease screening "
    "result for a patient. Use ONLY the facts given. Do not change or add any "
    "number. Do not diagnose. Do not recommend any treatment, medication, diet "
    "or test. Say clearly that this is a statistical estimate from a computer "
    "model, not a diagnosis, and suggest discussing it with a doctor. Write "
    "one paragraph of at most 110 words, in the second person, calm and clear."
)
BANNED = re.compile(r"\b(prescri\w*|statin\w*|aspirin|you should take|start taking|"
                    r"you have (?:coronary|heart disease)|diagnosed with)\b", re.I)


def fmt_pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def facts_text(clean: dict, pred: dict) -> str:
    up = [f["factor"].lower() for f in pred["factors"] if f["contribution"] > 0.05][:3]
    down = [f["factor"].lower() for f in pred["factors"] if f["contribution"] < -0.05][:3]
    status = ("at or above" if pred["flagged"] else "below")
    return (
        f"Estimated 10-year risk of a coronary heart disease event: {fmt_pct(pred['risk'])}.\n"
        f"This is {status} the screening threshold of {fmt_pct(pred['threshold'])} "
        f"used to flag patients for follow-up.\n"
        f"Average risk in the study population: {fmt_pct(pred['base_rate'])}.\n"
        f"Factors raising the estimate compared with an average patient: "
        f"{', '.join(up) or 'none notable'}.\n"
        f"Factors lowering the estimate: {', '.join(down) or 'none notable'}.\n"
        f"The model is a logistic regression trained on the Framingham Heart Study."
    )


def template_summary(pred: dict) -> str:
    up = [f["factor"].lower() for f in pred["factors"] if f["contribution"] > 0.05][:3]
    down = [f["factor"].lower() for f in pred["factors"] if f["contribution"] < -0.05][:2]
    s = (f"The model estimates a {fmt_pct(pred['risk'])} chance of a coronary heart "
         f"disease event in the next 10 years, compared with an average of "
         f"{fmt_pct(pred['base_rate'])} in the study population. This is "
         f"{'at or above' if pred['flagged'] else 'below'} the "
         f"{fmt_pct(pred['threshold'])} threshold used to flag patients for follow-up. ")
    if up:
        s += f"The factors that raised the estimate most were {', '.join(up)}. "
    if down:
        s += f"Factors that lowered it included {', '.join(down)}. "
    return s + ("This is a statistical estimate from a computer model, not a diagnosis. "
                "Please discuss it with a doctor.")


def check_summary(text: str, pred: dict) -> str | None:
    """Return a reason if the LLM text is unsafe or changes the numbers."""
    allowed = {fmt_pct(pred["risk"]), fmt_pct(pred["threshold"]), fmt_pct(pred["base_rate"])}
    for p in re.findall(r"\d+(?:\.\d+)?\s?%", text):
        if p.replace(" ", "") not in allowed:
            return f"contains a number not produced by the model ({p})"
    if fmt_pct(pred["risk"]) not in text.replace(" %", "%"):
        return "does not state the model's risk estimate"
    if BANNED.search(text):
        return "contains treatment or diagnostic wording"
    if len(text.split()) > 170:
        return "too long"
    return None


def ollama(path: str, payload: dict | None = None, timeout: float = 120):
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ---------------------------------------------------------------- routes
@app.get("/api/schema")
def schema():
    return {
        "fields": [{"name": n, "label": l, "group": g, "kind": k, "required": r,
                    "unit": u, "options": o,
                    "range": RANGES.get(n)} for n, l, g, k, r, u, o in FIELDS],
        "model": CARD["model"],
        "threshold": THRESHOLD,
        "base_rate": BASE_RATE,
        "validation": CARD["validation"],
        "llm_model": LLM_MODEL,
    }


@app.post("/api/predict")
def api_predict(raw: dict):
    clean, errors, warnings = validate(raw)
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)
    return {**predict(clean), "warnings": warnings}


@app.post("/api/summary")
def api_summary(raw: dict):
    clean, errors, _ = validate(raw)
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)
    pred = predict(clean)  # recomputed server-side: the LLM never sees client numbers
    t0 = time.time()
    try:
        r = ollama("/api/generate", {
            "model": LLM_MODEL, "system": SYSTEM_PROMPT,
            "prompt": facts_text(clean, pred), "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0.2, "num_predict": 220}})
        text = r["response"].strip()
        problem = check_summary(text, pred)
        if problem is None:
            return {"text": text, "source": "llm", "model": LLM_MODEL,
                    "seconds": round(time.time() - t0, 1)}
        reason = f"LLM text rejected: {problem}"
    except (urllib.error.URLError, TimeoutError, KeyError, OSError) as e:
        reason = f"local LLM unavailable ({type(e).__name__})"
    return {"text": template_summary(pred), "source": "template", "reason": reason,
            "seconds": round(time.time() - t0, 1)}


@app.get("/api/llm/status")
def llm_status():
    try:
        tags = ollama("/api/tags", timeout=3)
        names = [m["name"] for m in tags.get("models", [])]
        return {"online": True, "model": LLM_MODEL, "installed": LLM_MODEL in names}
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"online": False, "model": LLM_MODEL, "installed": False}


@app.post("/api/llm/warmup")
def llm_warmup():
    t0 = time.time()
    try:
        ollama("/api/generate", {"model": LLM_MODEL, "prompt": "Hello",
                                 "stream": False, "keep_alive": "30m",
                                 "options": {"num_predict": 1}}, timeout=180)
        return {"ready": True, "seconds": round(time.time() - t0, 1)}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ready": False, "error": type(e).__name__}


@app.get("/api/examples")
def examples():
    """Three dataset patients (validation set) with low, middle and high risk."""
    val = load_split("val").dropna(subset=FEATURES)
    val = val.assign(risk=MODEL.predict_proba(val[FEATURES])[:, 1]).sort_values("risk")
    picks = {"Lower-risk example": 0.06, "Average example": 0.15, "Higher-risk example": 0.40}
    out = []
    for name, target in picks.items():
        r = val.iloc[(val["risk"] - target).abs().argmin()]
        out.append({"name": name, "patient_id": int(r["patient_id"]),
                    "values": {f: (int(r[f]) if FIELD[f][3] in ("binary", "select")
                                   else float(r[f])) for f in FEATURES}})
    return out


@app.get("/api/comparison")
def comparison():
    comp = ROOT / "results" / "comparison"
    common = pd.read_csv(comp / "common.csv")
    overall = pd.read_csv(comp / "overall.csv")
    groups = pd.read_csv(comp / "by_group.csv")
    for df in (common, overall, groups):
        df["name"] = df["model"].map(MODEL_NAMES).fillna(df["model"])
    clean = lambda df: json.loads(df.to_json(orient="records"))  # noqa: E731
    return {"common": clean(common), "overall": clean(overall), "groups": clean(groups),
            "chosen": "logistic_regression"}


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
