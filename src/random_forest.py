#%% import
#model training and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, RocCurveDisplay, roc_auc_score)
#tools
import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

#others
from preprocessing import load_split, FEATURES, TARGET, build_preprocessor
from prepare_data import SEED

#%% load data and split into features and target for train, val, test
train_df = load_split("train")
x_train = train_df[FEATURES]
y_train = train_df[TARGET]
print("train - X:", x_train.shape, " y:", y_train.shape)

val_df = load_split("val")
x_val = val_df[FEATURES]
y_val = val_df[TARGET]
print("val - X:", x_val.shape, " y:", y_val.shape)

test_df = load_split("test")
x_test = test_df[FEATURES]
y_test = test_df[TARGET]
print("test - X:", x_test.shape, " y:", y_test.shape)

#train - X: (2966, 15)  y: (2966,)
#val - X: (636, 15)  y: (636,)
#test - X: (636, 15)  y: (636,)

#%% create a Random Forest model
print("Create RF Model")
rf_model = Pipeline([
    ("preprocessing", build_preprocessor(scale=False)),
    ("classifier", RandomForestClassifier(
        n_estimators=100, oob_score=False, random_state=SEED)
    )
])

#%% #region fit the model and evaluate on train
# rf_model.fit(x_train, y_train)
# train_probs = rf_model.predict_proba(x_train)[:, 1]
# print("Train ROC-AUC:", roc_auc_score(y_train, train_probs))

#%% #region Validation curve to find the best n_estimators
#Validation curve 1 and hyperparameter tuning for n_estimators
RUN_VALIDATION_CURVE = False
if RUN_VALIDATION_CURVE:
    lst_n_estimators = list(range(1, 201, 10)) #1, 11, 21, 31, 41, 51 -> 191
    scorel = []
    for i in lst_n_estimators:
        rf_model.set_params(classifier__n_estimators=i)

        score = cross_val_score(rf_model, x_train, y_train, cv=5, scoring="roc_auc")
        scorel.append(score.mean())

    #return the best n_estimators and mean ROC-AUC score
    best_n_estimators = lst_n_estimators[int(np.argmax(scorel))]
    print("Best n_estimators:", best_n_estimators, "Mean ROC-AUC:", max(scorel))
    # return : Best n_estimators: 181 Mean ROC-AUC: 0.6959307251727729

    # plt.figure(figsize=(20, 5))
    # plt.plot(lst_n_estimators, scorel, marker="x")
    # plt.xlabel("n_estimators")
    # plt.ylabel("Mean 5-fold ROC-AUC - 1")
    # plt.show()

    #Validation curve 2 and hyperparameter tuning for n_estimators
    min_arange = best_n_estimators - 10
    max_arange = best_n_estimators + 10
    print("min_arange:", min_arange, "max_arange:", max_arange)
    # return : min_arange: 171 max_arange: 191

    lst_n_estimators2 = list(range(min_arange, max_arange + 1, 1))
    scorel2 = []

    for i in range(min_arange, max_arange + 1, 1): 
        rf_model.set_params(classifier__n_estimators=i)

        score2 = cross_val_score(rf_model, x_train, y_train, cv=5, scoring="roc_auc")
        scorel2.append(score2.mean())

    #return the best n_estimators and mean ROC-AUC score
    best_n_estimators2 = lst_n_estimators2[int(np.argmax(scorel2))]
    print("Best n_estimators - 2:", best_n_estimators2, "Mean ROC-AUC - 2:", max(scorel2))
    # return : Best n_estimators: 182 Mean ROC-AUC: 0.6962303691081028

    # plt.figure(figsize=(20, 5))
    # plt.plot(lst_n_estimators2, scorel2, marker="x")
    # plt.xlabel("n_estimators")
    # plt.ylabel("Mean 5-fold ROC-AUC - 2")
    # plt.show()
#endregion

#%% cross-validation, grid search for hyperparameter tuning
# max_depth = np.arange(1, 21, 1) 
# max_features = np.arange(int(np.sqrt(len(FEATURES))), len(FEATURES) + 1, 1) #3 - 15
# max_samples = [0.2, 0.75, 1.0] 
print("GridSearch for hpyerparameters")
grid_search_cv = GridSearchCV(
    estimator = rf_model,
    param_grid = {
        "classifier__n_estimators" : [182], #Validated above
        "classifier__max_depth" : [4, 8, 12, None],
        "classifier__max_features" : ["sqrt", "log2", None],
        "classifier__min_samples_split" : [2, 5, 10]
    },
    cv = 10,
    scoring = "roc_auc"
)

grid_search_cv.fit(x_train, y_train)
# results = pd.DataFrame(grid_search_cv.cv_results_).to_string()
# print(results)
rf_model = grid_search_cv.best_estimator_
rf_model_params = grid_search_cv.best_params_
rf_model_score = grid_search_cv.best_score_
print("Best estimator: ", rf_model)
print("Best parameters: ", rf_model_params)
print("Best ROC-AUC score: ", rf_model_score)

#%% threshold
print("Validation threshold")
thresholds = [0.5] 
# thresholds = np.arange(0.1, 1.1, 0.05)
val_probs = rf_model.predict_proba(x_val)[:, 1] #column 0 + column 1 = 1.0

# best_threshold = None
# best_precision = -1
# min_recall = 0.7 #set minimum recall requirement

results = []
y_val_pred = []

for threshold in thresholds:
    y_val_pred = (val_probs >= threshold).astype(int)

    # calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, val_probs)
    
    # store results
    results.append({
        # "Threshold": round(threshold, 2),
        "Val Accuracy" : round(accuracy, 3),
        "Val Precision": round(precision, 3),
        "Val Recall": round(recall, 3),
        "Val F1-score" : round(f1, 3),
        "Val ROC_AUC" : round(roc_auc, 3)
    })

    # update best threshold if recall is above minimum and precision is better than previous best
    # if recall >= min_recall and precision > best_precision:
    #     best_precision = precision
    #     best_threshold = float(threshold)

best_threshold = float(threshold) #temp decision

df_results = pd.DataFrame(results)
print("Val set")
print(df_results.to_string(index=False)) #no index

# if best_threshold is None:
#     raise ValueError("No threshold achieved validation Recall >= 0.7.")

# print(confusion_matrix(y_val, y_val_pred))
ConfusionMatrixDisplay.from_predictions(
    y_val, y_val_pred, display_labels=["No TenYearCHD", "TenYearCHD"], cmap="Blues", colorbar=False
)
plt.title("Random Forest: confusion matrix on val set")
plt.grid(False)
plt.show()

#TODO: Youden's J statistic or F2-score

#0.100
#[[233 306]
# [ 12  85]]
#0.5
#[[538   1]
 #[ 94   3]]

#%% Test
test_probs = rf_model.predict_proba(x_test)[:, 1]
y_test_pred = (test_probs >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, test_probs)
cm = confusion_matrix(y_test, y_test_pred)

print("Test set")
# print(f"Threshold: {best_threshold:.2f}")
print(f"Test Accuracy: {accuracy:.3f}")
print(f"Test Precision: {precision:.3f}")
print(f"Test Recall: {recall:.3f}")
print(f"Test F1-score: {f1:.3f}")
print(f"Test ROC-AUC: {roc_auc:.3f}")

# print(cm)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred, display_labels=["No TenYearCHD", "TenYearCHD"], cmap="Blues", colorbar=False
)
plt.title("Random Forest: confusion matrix on test set")
plt.grid(False)
plt.show()

#%% output ROC curve and confusion matrix
from pathlib import Path
print("Ouput Roc curve and cm")
output_dir = Path(__file__).resolve().parents[1] / "outputs" / "random_forest" / "chart"
output_dir.mkdir(parents=True, exist_ok=True)

#roc curve
fig, ax = plt.subplots(figsize=(9, 7))
RocCurveDisplay.from_predictions(
    y_test,
    test_probs,
    name="Random Forest",
    ax=ax
)
ax.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
ax.set_title("Random Forest ROC Curve")
ax.legend(loc="lower right")
fig.savefig(output_dir / "roc_curve_test.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

#confusion matrix
fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    labels=[0, 1],
    display_labels=["No TenYearCHD", "TenYearCHD"],
    cmap="Blues",
    ax=ax
)
ax.set_title(f"Confusion Matrix (Test Set, threshold={best_threshold:.2f})")
fig.savefig(output_dir / "confusion_matrix_test.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

#TODO: Performance Across Patient Groups


#%% build json report for test results
import json
import pprint

print("output json report")
output_dir = Path(__file__).resolve().parents[1] / "outputs" / "random_forest" / "test_results"
output_dir.mkdir(parents=True, exist_ok=True)

best_params = {
    key: value.item() if isinstance(value, np.generic) else value
    for key, value in rf_model_params.items()
}

json_report = {
    "model_name": "Random Forest",
    "cv": {
        "folds": 10,
        "scoring": "roc_auc",
        "best_mean_roc_auc": float(rf_model_score),
    },
    "best_params": best_params,
    # "threshold_selection": {
    #     "method": "Highest precision among thresholds with validation recall >= target",
    #     "validation_recall_target": float(min_recall),
    #     "selected_threshold": float(best_threshold),
    # },
    "threshold": float(best_threshold),
    "test_metrics": {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "roc_auc": round(roc_auc, 3),
        "confusion_matrix": cm.tolist()
    },
}

with open(output_dir / "test_results.json", "w", encoding="utf-8") as file:
    json.dump(json_report, file, indent=4)

pprint.pprint(json_report, sort_dicts=False)

#create a CSV file with patient_id, true label, predicted label, predicted probability
predictions_df = test_df[["patient_id"]].copy()
predictions_df["true_label"] = y_test.to_numpy()
predictions_df["predicted_label"] = y_test_pred
predictions_df["predicted_probability"] = np.round(test_probs, 3)

predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

#%%
import joblib
model_dir = Path(__file__).resolve().parents[1] / "outputs" / "random_forest" / "model"
model_dir.mkdir(parents=True, exist_ok=True)

print("output model")
joblib.dump(
    {
        "model": rf_model,
        "threshold": best_threshold,
    },
    model_dir / "random_forest.joblib",
)