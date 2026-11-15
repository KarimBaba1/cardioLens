"""Framingham Risk Score baseline (BMI-based, no lipids needed).

Run:  python src/framingham_score.py

Formula: D'Agostino et al., "General Cardiovascular Risk Profile for Use in
Primary Care", Circulation 2008;117:743-753 (simple office-based model).
    risk_10y = 1 - S0 ** exp( sum(beta * x) - mean )

Coefficients cross-checked against two independent sources:
  1. R package CVrisk, data-raw/score_coef.R (object frs_simple_coef)
     https://raw.githubusercontent.com/vcastro/CVrisk/master/data-raw/score_coef.R
  2. Table 2 of Unnikrishnan et al. 2016, Comput Math Methods Med
     https://pmc.ncbi.nlm.nih.gov/articles/PMC4993959/
NOT yet checked against the original Circulation article (could not be opened).

LIMITATIONS (state these in the report):
  * It predicts general cardiovascular disease (CHD, stroke, peripheral artery
    disease, heart failure), which is broader than TenYearCHD.
  * The BMI model is not CHD-specific and is not calibrated to this dataset;
    probabilities overstate CHD risk. Use ROC-AUC for discrimination.
  * Patients missing any required input are NOT filled in: they are left out
    of results/framingham_score/predictions.csv and counted in the report.
  * The classification threshold is chosen on the validation set (best F1),
    like every other model. No test labels are used for that choice.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from preprocessing import TARGET, load_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "framingham_score"

COEF = {
    "male": dict(age=3.11296, bmi=0.79277, sbp_untreated=1.85508,
                 sbp_treated=1.92672, smoker=0.70953, diabetes=0.53160,
                 mean=23.9388, s0=0.88431),
    "female": dict(age=2.72107, bmi=0.51125, sbp_untreated=2.81291,
                   sbp_treated=2.88267, smoker=0.61868, diabetes=0.77763,
                   mean=26.0145, s0=0.94833),
}
REQUIRED = ["male", "age", "sysBP", "BPMeds", "currentSmoker", "diabetes", "BMI"]


def frs_risk(df: pd.DataFrame) -> pd.Series:
    """10-year general CVD risk (0-1). Rows with missing inputs -> NaN."""
    risk = pd.Series(np.nan, index=df.index)
    complete = df[REQUIRED].notna().all(axis=1)
    for sex, c in (("male", COEF["male"]), ("female", COEF["female"])):
        rows = df[complete & (df["male"] == (1 if sex == "male" else 0))]
        sbp_beta = np.where(rows["BPMeds"] == 1, c["sbp_treated"], c["sbp_untreated"])
        total = (c["age"] * np.log(rows["age"])
                 + c["bmi"] * np.log(rows["BMI"])
                 + sbp_beta * np.log(rows["sysBP"])
                 + c["smoker"] * rows["currentSmoker"]
                 + c["diabetes"] * rows["diabetes"])
        risk.loc[rows.index] = 1 - c["s0"] ** np.exp(total - c["mean"])
    return risk


def best_f1_threshold(y, p) -> float:
    grid = np.round(np.arange(0.02, 0.80, 0.01), 2)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def main() -> None:
    val, test = load_split("val"), load_split("test")
    val["risk"], test["risk"] = frs_risk(val), frs_risk(test)

    v = val.dropna(subset=["risk"])
    threshold = best_f1_threshold(v[TARGET], v["risk"])

    t = test.dropna(subset=["risk"])
    pred = (t["risk"] >= threshold).astype(int)
    result = pd.DataFrame({
        "patient_id": t["patient_id"].to_numpy(),
        "actual": t[TARGET].to_numpy(),
        "predicted": pred.to_numpy(),
        "probability": t["risk"].to_numpy(),
    })
    metrics = pd.DataFrame([{
        "model": "Framingham Risk Score (BMI)",
        "threshold": threshold,
        "n_test_scored": len(t),
        "n_test_excluded_missing_inputs": len(test) - len(t),
        "accuracy": float((pred == t[TARGET]).mean()),
        "precision": precision_score(t[TARGET], pred, zero_division=0),
        "recall": recall_score(t[TARGET], pred, zero_division=0),
        "f1_score": f1_score(t[TARGET], pred, zero_division=0),
        "roc_auc": roc_auc_score(t[TARGET], t["risk"]),
    }])

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "predictions.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    print(f"Validation: {len(v)} scored, threshold (best F1) = {threshold}")
    print(f"Test: {len(t)} scored, {len(test) - len(t)} excluded (missing inputs)")
    print(f"Mean predicted 10y risk: {t['risk'].mean():.3f} "
          f"vs observed CHD rate {t[TARGET].mean():.3f}")
    print(metrics.drop(columns="model").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
