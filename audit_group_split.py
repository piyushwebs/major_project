
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from preprocess import sequences, y

# Assign one integer group ID to each unique complete sequence.
group_lookup = {}
groups = np.empty(len(sequences), dtype=np.int64)

for i, sequence in enumerate(sequences):
    key = tuple(sequence)

    if key not in group_lookup:
        group_lookup[key] = len(group_lookup)

    groups[i] = group_lookup[key]

print(f"Total rows: {len(sequences):,}")
print(f"Unique complete-sequence groups: {len(group_lookup):,}")

# Create 20 group-aware folds.
splitter = StratifiedGroupKFold(
    n_splits=20,
    shuffle=True,
    random_state=42
)

folds = list(
    splitter.split(np.zeros(len(y)), y, groups)
)

# 70% train, 15% validation, 15% test.
train_idx = np.concatenate([folds[i][1] for i in range(14)])
val_idx = np.concatenate([folds[i][1] for i in range(14, 17)])
test_idx = np.concatenate([folds[i][1] for i in range(17, 20)])

splits = {
    "train": train_idx,
    "validation": val_idx,
    "test": test_idx,
}

print("\n=== GROUP-AWARE SPLIT AUDIT ===")

for name, indices in splits.items():
    labels = y[indices]
    normal = int(np.sum(labels == 0))
    anomaly = int(np.sum(labels == 1))

    print(f"\n{name.upper()}")
    print(f"Rows: {len(indices):,}")
    print(f"Normal: {normal:,}")
    print(f"Anomaly: {anomaly:,}")
    print(f"Anomaly percentage: {100 * anomaly / len(indices):.3f}%")

# Check complete-sequence groups across splits.
group_sets = {
    name: set(groups[indices])
    for name, indices in splits.items()
}

print("\n=== CROSS-SPLIT GROUP OVERLAP ===")
print("Train-validation:", len(group_sets["train"] & group_sets["validation"]))
print("Train-test:", len(group_sets["train"] & group_sets["test"]))
print("Validation-test:", len(group_sets["validation"] & group_sets["test"]))

# Also check the actual MAX_LEN=50 inputs used by the model.
from preprocess import X

encoded_sets = {
    name: {X[i].tobytes() for i in indices}
    for name, indices in splits.items()
}

print("\n=== ENCODED INPUT OVERLAP (MAX_LEN=50) ===")
print("Train-validation:", len(encoded_sets["train"] & encoded_sets["validation"]))
print("Train-test:", len(encoded_sets["train"] & encoded_sets["test"]))
print("Validation-test:", len(encoded_sets["validation"] & encoded_sets["test"]))

print("\nAudit complete. No model training performed.")
