"""Shared loading + model-specific preprocessing for CardioLens.

Usage (e.g. Random Forest):
    from preprocessing import load_split, FEATURES, TARGET, build_preprocessor
    train = load_split("train")
    pre = build_preprocessor(scale=False)        # RF: impute only
    X_train = pre.fit_transform(train[FEATURES]) # fit on TRAIN ONLY
    X_val = pre.transform(load_split("val")[FEATURES])

Put the preprocessor inside an sklearn Pipeline with the model so it is
always fitted on training data only (including inside cross-validation).
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

TARGET = "TenYearCHD"
BINARY_FEATURES = ["male", "currentSmoker", "BPMeds", "prevalentStroke",
                   "prevalentHyp", "diabetes"]
NUMERIC_FEATURES = ["age", "education", "cigsPerDay", "totChol", "sysBP",
                    "diaBP", "BMI", "heartRate", "glucose"]
FEATURES = BINARY_FEATURES + NUMERIC_FEATURES


def load_split(name: str) -> pd.DataFrame:
    """name: 'train', 'val', 'test' or 'llm_test_subset'."""
    return pd.read_csv(PROCESSED / f"{name}.csv")


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """scale=True for Logistic Regression, False for Random Forest."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
        ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
    ])


def row_to_text(row: pd.Series) -> str:
    """Patient description for LLM prompts. Never includes the label."""
    def v(col, fmt="{}", unknown="unknown"):
        return unknown if pd.isna(row[col]) else fmt.format(row[col])

    yn = lambda col: v(col, unknown="unknown") if pd.isna(row[col]) else (
        "yes" if row[col] == 1 else "no")
    return (
        f"Sex: {'male' if row['male'] == 1 else 'female'}; "
        f"Age: {v('age', '{:.0f}')} years; "
        f"Education level (1-4): {v('education', '{:.0f}')}; "
        f"Current smoker: {yn('currentSmoker')}; "
        f"Cigarettes per day: {v('cigsPerDay', '{:.0f}')}; "
        f"On blood pressure medication: {yn('BPMeds')}; "
        f"Previous stroke: {yn('prevalentStroke')}; "
        f"Hypertensive: {yn('prevalentHyp')}; "
        f"Diabetes: {yn('diabetes')}; "
        f"Total cholesterol: {v('totChol', '{:.0f}')} mg/dL; "
        f"Systolic BP: {v('sysBP', '{:.1f}')} mmHg; "
        f"Diastolic BP: {v('diaBP', '{:.1f}')} mmHg; "
        f"BMI: {v('BMI', '{:.2f}')}; "
        f"Heart rate: {v('heartRate', '{:.0f}')} bpm; "
        f"Glucose: {v('glucose', '{:.0f}')} mg/dL"
    )
