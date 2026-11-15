"""Train and save the final CardioLens model used by the web app.

Run:  python src/train_final_model.py

Logistic Regression was chosen because it was tied for the best ROC-AUC in
the model comparison, is the simplest approach, and its coefficients give an
exact per-patient explanation. Settings are identical to
src/logistic_regression.py (trained on train.csv, threshold = best F1 on
val.csv). The test set is not touched here.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from preprocessing import FEATURES, NUMERIC_FEATURES, TARGET, build_preprocessor, load_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"


def main() -> None:
    train, val = load_split("train"), load_split("val")
    model = Pipeline([
        ("preprocessing", build_preprocessor(scale=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]).fit(train[FEATURES], train[TARGET])

    val_p = model.predict_proba(val[FEATURES])[:, 1]
    grid = [round(float(t), 2) for t in np.arange(0.05, 0.95, 0.05)]
    f1s = [f1_score(val[TARGET], (val_p >= t).astype(int), zero_division=0) for t in grid]
    threshold = grid[int(np.argmax(f1s))]
    val_pred = (val_p >= threshold).astype(int)

    pre = model.named_steps["preprocessing"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    train_t = pre.transform(train[FEATURES])

    card = {
        "model": "Logistic Regression",
        "threshold": threshold,
        "base_rate": float(train[TARGET].mean()),
        "feature_order": names,
        "train_mean_transformed": train_t.mean(axis=0).tolist(),
        # the model only saw values in these ranges; inputs outside are rejected
        "supported_ranges": {c: [float(train[c].min()), float(train[c].max())]
                             for c in NUMERIC_FEATURES},
        # 98% of training patients fall inside these; values outside are rare,
        # so the app warns that the estimate is less reliable there
        "typical_ranges": {c: [float(train[c].quantile(0.01)), float(train[c].quantile(0.99))]
                           for c in NUMERIC_FEATURES},
        "validation": {
            "roc_auc": float(roc_auc_score(val[TARGET], val_p)),
            "precision": float(precision_score(val[TARGET], val_pred, zero_division=0)),
            "recall": float(recall_score(val[TARGET], val_pred, zero_division=0)),
            "f1": float(f1_score(val[TARGET], val_pred, zero_division=0)),
        },
        "n_train": len(train),
    }
    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "logistic_regression.joblib")
    (OUT / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps({k: card[k] for k in ("threshold", "base_rate", "validation")}, indent=2))


if __name__ == "__main__":
    main()
