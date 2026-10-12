
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

RANDOM_STATE = 42

# --------------------------------------------------
# Recreate group-aware split
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

texts = [
    " ".join(f"E{event_id}" for event_id in sequence[:50])
    for sequence in preprocess.sequences
]

# --------------------------------------------------
# Compare two feature configurations
# --------------------------------------------------
configurations = [
    ("Unigrams only", (1, 1)),
    ("Unigrams + bigrams", (1, 2)),
]

results = []

for name, ngram_range in configurations:
    print("\n" + "=" * 60)
    print("Configuration:", name)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngram_range,
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

    print("Number of features:", X_train.shape[1])

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    model.fit(X_train, y[train_idx])

    # Select threshold using validation data only.
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

    # Evaluate the test set once at the selected threshold.
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    precision = precision_score(
        y[test_idx], test_preds, zero_division=0
    )
    recall = recall_score(
        y[test_idx], test_preds, zero_division=0
    )
    f1 = f1_score(
        y[test_idx], test_preds, zero_division=0
    )
    pr_auc = average_precision_score(y[test_idx], test_probs)
    roc_auc = roc_auc_score(y[test_idx], test_probs)
    matrix = confusion_matrix(y[test_idx], test_preds)

    results.append({
        "configuration": name,
        "features": X_train.shape[1],
        "threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
    })

    print("Selected threshold:", round(best_threshold, 4))
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1 score:  {f1:.6f}")
    print(f"PR-AUC:    {pr_auc:.6f}")
    print(f"ROC-AUC:   {roc_auc:.6f}")
    print("Confusion matrix:")
    print(matrix)

# --------------------------------------------------
# Final comparison
# --------------------------------------------------
print("\n" + "=" * 75)
print("=== ABLATION COMPARISON ===")

for result in results:
    print(
        f"\n{result['configuration']}\n"
        f"Features: {result['features']}\n"
        f"F1: {result['f1']:.6f} | "
        f"Precision: {result['precision']:.6f} | "
        f"Recall: {result['recall']:.6f}\n"
        f"PR-AUC: {result['pr_auc']:.6f} | "
        f"ROC-AUC: {result['roc_auc']:.6f}\n"
        f"False positives: {result['false_positives']} | "
        f"False negatives: {result['false_negatives']}"
    )

print("\nAblation experiment complete.")
