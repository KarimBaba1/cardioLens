"""Shared evaluation for CardioLens (model comparison + patient-group view).

Run:  python src/evaluate.py

Every approach saves results/<name>/predictions.csv with the columns
    patient_id, actual, predicted, probability
(probability may be empty for approaches without one, e.g. an LLM that only
gives a label; ROC-AUC is then reported as blank).

Outputs (results/comparison/):
    overall.csv    metrics per model on its own test patients
    common.csv     metrics per model on the patients ALL models share
                   (use this table for the fair comparison; the LLM runs
                   on a 300-patient subset)
    by_group.csv   metrics by sex and age group, with n and positives
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from preprocessing import load_split

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "comparison"

MIN_POSITIVES = 20  # below this a group's metrics are flagged unreliable
AGE_BINS = [0, 44, 54, 200]
AGE_LABELS = ["<45", "45-54", "55+"]


def metrics(df: pd.DataFrame) -> dict:
    y, pred = df["actual"], df["predicted"]
    has_prob = df["probability"].notna().all() and y.nunique() == 2
    return {
        "n": len(df),
        "positives": int(y.sum()),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, df["probability"]) if has_prob else None,
    }


def load_models() -> dict[str, pd.DataFrame]:
    models = {}
    for pred_file in sorted(RESULTS.glob("*/predictions.csv")):
        name = pred_file.parent.name
        if name == "comparison":
            continue
        df = pd.read_csv(pred_file)
        thr_file = pred_file.parent / "metrics.csv"
        thr = (pd.read_csv(thr_file)["threshold"].iloc[0]
               if thr_file.exists() and "threshold" in pd.read_csv(thr_file)
               else None)
        df.attrs["threshold"] = thr
        models[name] = df
    return models


def main() -> None:
    models = load_models()
    if not models:
        raise SystemExit("No results/*/predictions.csv found")
    OUT.mkdir(parents=True, exist_ok=True)

    test = load_split("test")[["patient_id", "male", "age"]]
    test["age_group"] = pd.cut(test["age"], bins=AGE_BINS, labels=AGE_LABELS)
    test["sex"] = test["male"].map({1: "male", 0: "female"})

    # Sanity: predictions must refer to real test patients, labels must match
    truth = load_split("test").set_index("patient_id")["TenYearCHD"]
    for name, df in models.items():
        assert df["patient_id"].isin(truth.index).all(), f"{name}: unknown patient_id"
        assert (df.set_index("patient_id")["actual"] == truth.loc[df["patient_id"]]
                ).all(), f"{name}: labels differ from test.csv"

    overall = pd.DataFrame(
        [{"model": n, "threshold": d.attrs["threshold"], **metrics(d)}
         for n, d in models.items()])
    overall.to_csv(OUT / "overall.csv", index=False)

    common_ids = set.intersection(*(set(d["patient_id"]) for d in models.values()))
    common = pd.DataFrame(
        [{"model": n, "threshold": d.attrs["threshold"],
          **metrics(d[d["patient_id"].isin(common_ids)])}
         for n, d in models.items()])
    common.to_csv(OUT / "common.csv", index=False)

    rows = []
    for name, df in models.items():
        merged = df.merge(test, on="patient_id")
        for dim in ("sex", "age_group"):
            for group, g in merged.groupby(dim, observed=True):
                m = metrics(g)
                rows.append({"model": name, "dimension": dim, "group": str(group),
                             **m,
                             "reliable": m["positives"] >= MIN_POSITIVES})
    by_group = pd.DataFrame(rows)
    by_group.to_csv(OUT / "by_group.csv", index=False)

    pd.set_option("display.width", 200)
    fmt = {"float_format": lambda v: f"{v:.3f}"}
    print(f"\n== Overall (each model on its own test patients) ==")
    print(overall.to_string(index=False, **fmt))
    print(f"\n== Common patients across all models (n={len(common_ids)}) ==")
    print(common.to_string(index=False, **fmt))
    print("\n== By patient group ==")
    print(by_group.to_string(index=False, **fmt))
    print(f"\n'reliable' = at least {MIN_POSITIVES} positive cases in the group. "
          "Group results show uneven performance; they do not prove clinical fairness.")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
