
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

import preprocess

# --------------------------------------------------
# Recreate the same group-aware split
# --------------------------------------------------
group_lookup = {}
groups = np.empty(len(preprocess.sequences), dtype=np.int64)

for i, sequence in enumerate(preprocess.sequences):
    key = tuple(sequence)
    if key not in group_lookup:
        group_lookup[key] = len(group_lookup)
    groups[i] = group_lookup[key]

splitter = StratifiedGroupKFold(
    n_splits=20,
    shuffle=True,
    random_state=42,
)

folds = list(
    splitter.split(
        np.zeros(len(preprocess.y)),
        preprocess.y,
        groups,
    )
)

train_idx = np.concatenate([folds[i][1] for i in range(14)])
val_idx = np.concatenate([folds[i][1] for i in range(14, 17)])
test_idx = np.concatenate([folds[i][1] for i in range(17, 20)])

assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0

# --------------------------------------------------
# Use ONLY original sequence length as the feature
# --------------------------------------------------
X_length = np.array(
    [len(sequence) for sequence in preprocess.sequences],
    dtype=np.float64,
).reshape(-1, 1)

y = np.asarray(preprocess.y, dtype=int)

X_train, y_train = X_length[train_idx], y[train_idx]
X_val, y_val = X_length[val_idx], y[val_idx]
X_test, y_test = X_length[test_idx], y[test_idx]

# Train only on the training partition.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)

model.fit(X_train, y_train)

# Select a threshold using validation data only.
val_probs = model.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.01, 0.99, 99)

best_threshold = max(
    thresholds,
    key=lambda t: f1_score(
        y_val,
        (val_probs >= t).astype(int),
        zero_division=0,
    ),
)

# Evaluate the selected threshold on the untouched test set.
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

print("\n=== LENGTH-ONLY BASELINE ===")
print("Features used: original sequence length only")
print("Train samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Test samples:", len(test_idx))
print("Validation-selected threshold:", round(best_threshold, 4))

print(f"Test precision: {precision_score(y_test, test_preds, zero_division=0):.6f}")
print(f"Test recall:    {recall_score(y_test, test_preds, zero_division=0):.6f}")
print(f"Test F1:        {f1_score(y_test, test_preds, zero_division=0):.6f}")
print(f"Test PR-AUC:    {average_precision_score(y_test, test_probs):.6f}")
print(f"Test ROC-AUC:   {roc_auc_score(y_test, test_probs):.6f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_preds))

print("\n=== ANOMALY RATE BY ORIGINAL LENGTH ===")
lengths = X_length[:, 0]

for lower, upper in [(0, 10), (10, 20), (20, 30), (30, 50), (50, float("inf"))]:
    mask = (lengths > lower) & (lengths <= upper)
    if mask.any():
        print(
            f"Length ({lower}, {upper}]: "
            f"samples={int(mask.sum())}, "
            f"anomalies={int(y[mask].sum())}, "
            f"anomaly_rate={y[mask].mean():.4%}"
        )

print("\nLength-only experiment complete.")
