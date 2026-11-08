"""Shared data preparation for CardioLens.

Run once:  python src/prepare_data.py

Produces fixed train/val/test splits that EVERY approach (LR, RF, LLM, FRS)
must use, so the comparison is fair. It deliberately does NOT impute or scale:
those steps are learned, so each model must fit them on train only
(see src/preprocessing.py).
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "framingham.csv"
OUT = ROOT / "data" / "processed"

SEED = 42
TARGET = "TenYearCHD"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
LLM_SUBSET_SIZE = 300  # fixed stratified subset of TEST for LLM runs

BINARY = ["male", "currentSmoker", "BPMeds", "prevalentStroke",
          "prevalentHyp", "diabetes", TARGET]
# Plausible physiological ranges; values outside are flagged as invalid -> NA.
RANGES = {
    "age": (20, 100),
    "education": (1, 4),
    "cigsPerDay": (0, 100),
    "totChol": (80, 700),
    "sysBP": (70, 300),
    "diaBP": (30, 200),
    "BMI": (10, 70),
    "heartRate": (30, 220),
    "glucose": (30, 500),
}
# Inputs needed by the BMI-based Framingham score (D'Agostino 2008).
FRS_INPUTS = ["male", "age", "sysBP", "BPMeds", "currentSmoker", "diabetes", "BMI"]


def load_and_validate() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW, na_values=["NA", ""])
    report = {"raw_rows": len(df), "raw_columns": list(df.columns)}

    expected = list(RANGES) + BINARY
    missing_cols = set(expected) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Exact duplicate rows
    dups = int(df.duplicated().sum())
    report["duplicate_rows_dropped"] = dups
    df = df.drop_duplicates().reset_index(drop=True)

    # Target must be complete and binary
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("TenYearCHD must be complete and 0/1")

    # Binary columns must be 0/1 (NA allowed except target)
    invalid = {}
    for col in BINARY:
        bad = df[col].notna() & ~df[col].isin([0, 1])
        if bad.any():
            invalid[col] = int(bad.sum())
            df.loc[bad, col] = pd.NA

    # Out-of-range numeric values -> NA (treated as missing, never silently kept)
    for col, (lo, hi) in RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        if bad.any():
            invalid[col] = int(bad.sum())
            df.loc[bad, col] = pd.NA
    report["out_of_range_set_to_na"] = invalid

    # Logical consistency: non-smokers should smoke 0 cigs/day
    incon = (df["currentSmoker"] == 0) & (df["cigsPerDay"] > 0)
    report["nonsmoker_with_cigs_fixed"] = int(incon.sum())
    df.loc[incon, "cigsPerDay"] = 0
    # Smokers with missing cigsPerDay stay NA (imputed later, train-fitted).

    # Stable ID so every team member can trace the same patient
    df.insert(0, "patient_id", range(1, len(df) + 1))

    report["rows_after_cleaning"] = len(df)
    report["missing_per_column"] = {
        c: int(n) for c, n in df.isna().sum().items() if n > 0}
    report["rows_with_any_missing"] = int(df.isna().any(axis=1).sum())
    report["rows_missing_frs_inputs"] = int(df[FRS_INPUTS].isna().any(axis=1).sum())
    return df, report


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=train_val[TARGET],
        random_state=SEED)
    llm_subset, _ = train_test_split(
        test, train_size=LLM_SUBSET_SIZE, stratify=test[TARGET],
        random_state=SEED)
    return {"train": train, "val": val, "test": test,
            "llm_test_subset": llm_subset}


def main() -> None:
    df, report = load_and_validate()
    parts = split(df)

    OUT.mkdir(parents=True, exist_ok=True)
    report["splits"] = {}
    for name, part in parts.items():
        part = part.sort_values("patient_id")
        part.to_csv(OUT / f"{name}.csv", index=False)
        report["splits"][name] = {
            "rows": len(part),
            "positives": int(part[TARGET].sum()),
            "positive_rate": round(float(part[TARGET].mean()), 4),
        }

    # Sanity: no patient appears in more than one of train/val/test
    ids = [set(parts[n]["patient_id"]) for n in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set(parts["llm_test_subset"]["patient_id"]) <= ids[2]

    report["seed"] = SEED
    (OUT / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
