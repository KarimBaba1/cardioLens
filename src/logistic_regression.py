
from pathlib import Path

import pandas as pd
from preprocessing import load_split, FEATURES, TARGET, build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

print("CardioLens Logistic Regression started")


# Load training and validation datasets
train = load_split("train")
val = load_split("val")


# Set up logistic regression with preprocessing
model = Pipeline([
    ("preprocessing", build_preprocessor(scale=True)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Train the model
model.fit(train[FEATURES], train[TARGET])


# Evaluate the model on validation data
predictions = model.predict(val[FEATURES])
probabilities = model.predict_proba(val[FEATURES])[:, 1]

print("\n--- Validation Results ---")
print("Accuracy:", accuracy_score(val[TARGET], predictions))
print("Precision:", precision_score(val[TARGET], predictions, zero_division=0))
print("Recall:", recall_score(val[TARGET], predictions, zero_division=0))
print("F1-score:", f1_score(val[TARGET], predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(val[TARGET], probabilities))


# Compare different thresholds using validation data
print("\n--- Threshold Comparison ---")

for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
    new_predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        val[TARGET], new_predictions, zero_division=0
    )
    recall = recall_score(
        val[TARGET], new_predictions, zero_division=0
    )
    f1 = f1_score(
        val[TARGET], new_predictions, zero_division=0
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )


# Test balanced logistic regression to handle class imbalance
print("\n--- Balanced Logistic Regression ---")

balanced_model = Pipeline([
    ("preprocessing", build_preprocessor(scale=True)),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

balanced_model.fit(train[FEATURES], train[TARGET])

balanced_probabilities = balanced_model.predict_proba(
    val[FEATURES]
)[:, 1]

print("Balanced ROC-AUC:",
      roc_auc_score(val[TARGET], balanced_probabilities))

for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
    pred = (balanced_probabilities >= threshold).astype(int)

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision_score(val[TARGET], pred, zero_division=0):.3f} | "
        f"Recall: {recall_score(val[TARGET], pred, zero_division=0):.3f} | "
        f"F1: {f1_score(val[TARGET], pred, zero_division=0):.3f}"
    )


# Final evaluation using the shared test dataset
print("\n--- FINAL TEST EVALUATION ---")

test = load_split("test")

# Use the normal logistic regression model
test_probabilities = model.predict_proba(
    test[FEATURES]
)[:, 1]

# Threshold 0.2 was selected based on validation F1-score
selected_threshold = 0.2

test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

print("Test Accuracy:",
      accuracy_score(test[TARGET], test_predictions))

print("Test Precision:",
      precision_score(test[TARGET], test_predictions, zero_division=0))

print("Test Recall:",
      recall_score(test[TARGET], test_predictions, zero_division=0))

print("Test F1-score:",
      f1_score(test[TARGET], test_predictions, zero_division=0))

print("Test ROC-AUC:",
      roc_auc_score(test[TARGET], test_probabilities))

print("\nConfusion Matrix:")
print(confusion_matrix(test[TARGET], test_predictions))

print("\nClassification Report:")
print(classification_report(
    test[TARGET], test_predictions, zero_division=0
))


# Create a folder to save results
project_root = Path(__file__).resolve().parents[1]
results_folder = project_root / "results" / "logistic_regression"
results_folder.mkdir(parents=True, exist_ok=True)


# Save test predictions
results = pd.DataFrame({
    "patient_id": test["patient_id"].to_numpy(),
    "actual": test[TARGET].to_numpy(),
    "predicted": test_predictions,
    "probability": test_probabilities
})

output_file = results_folder / "predictions.csv"
results.to_csv(output_file, index=False)

print("\nPredictions saved to:", output_file)


# Save test evaluation metrics
metrics = pd.DataFrame([{
    "model": "Logistic Regression",
    "threshold": selected_threshold,
    "accuracy": accuracy_score(test[TARGET], test_predictions),
    "precision": precision_score(
        test[TARGET], test_predictions, zero_division=0
    ),
    "recall": recall_score(
        test[TARGET], test_predictions, zero_division=0
    ),
    "f1_score": f1_score(
        test[TARGET], test_predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(test[TARGET], test_probabilities)
}])

metrics_file = results_folder / "metrics.csv"
metrics.to_csv(metrics_file, index=False)

print("Metrics saved to:", metrics_file)
