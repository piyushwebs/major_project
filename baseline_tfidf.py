
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
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
# Configuration
# --------------------------------------------------
RANDOM_STATE = 42

# --------------------------------------------------
# Recreate the exact group-aware split
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
    random_state=RANDOM_STATE,
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

assert len(set(groups[train_idx]) & set(groups[val_idx])) == 0
assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0
assert len(set(groups[val_idx]) & set(groups[test_idx])) == 0

y = np.asarray(preprocess.y, dtype=int)

# Convert event IDs into text tokens.
# Example: [5, 22, 11] -> "E5 E22 E11"
texts = [
    " ".join(f"E{event_id}" for event_id in sequence[:50])
    for sequence in preprocess.sequences
]

print("Train samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Test samples:", len(test_idx))

# --------------------------------------------------
# Fit TF-IDF on training data only
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b",
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=10000,
)

X_train = vectorizer.fit_transform(
    [texts[i] for i in train_idx]
)
X_val = vectorizer.transform(
    [texts[i] for i in val_idx]
)
X_test = vectorizer.transform(
    [texts[i] for i in test_idx]
)

print("TF-IDF feature count:", X_train.shape[1])

# --------------------------------------------------
# Train Logistic Regression
# --------------------------------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    solver="liblinear",
)

model.fit(X_train, y[train_idx])

# --------------------------------------------------
# Select threshold on validation data only
# --------------------------------------------------
val_probs = model.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.01, 0.99, 99)

best_threshold = max(
    thresholds,
    key=lambda threshold: f1_score(
        y[val_idx],
        (val_probs >= threshold).astype(int),
        zero_division=0,
    ),
)

# --------------------------------------------------
# Final evaluation on the untouched test set
# --------------------------------------------------
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

print("\n=== TF-IDF + LOGISTIC REGRESSION RESULTS ===")
print("Selected threshold:", round(best_threshold, 4))
print("Test samples:", len(test_idx))
print("Test anomalies:", int(y[test_idx].sum()))

print(
    f"Precision: {precision_score(y[test_idx], test_preds, zero_division=0):.6f}"
)
print(
    f"Recall:    {recall_score(y[test_idx], test_preds, zero_division=0):.6f}"
)
print(
    f"F1 score:  {f1_score(y[test_idx], test_preds, zero_division=0):.6f}"
)
print(
    f"PR-AUC:    {average_precision_score(y[test_idx], test_probs):.6f}"
)
print(
    f"ROC-AUC:   {roc_auc_score(y[test_idx], test_probs):.6f}"
)

print("\nConfusion matrix:")
print(confusion_matrix(y[test_idx], test_preds))

print("\nTF-IDF baseline evaluation complete.")
