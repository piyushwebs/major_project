
from collections import defaultdict
import numpy as np

from preprocess import (
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

splits = {
    "train": (X_train, y_train),
    "validation": (X_val, y_val),
    "test": (X_test, y_test),
}

# Map each encoded sequence to the labels observed in each split.
sequence_labels = defaultdict(lambda: defaultdict(set))

for split_name, (X, y) in splits.items():
    for sequence, label in zip(X, y):
        key = sequence.tobytes()
        sequence_labels[key][split_name].add(int(label))

print("\n=== ROW-LEVEL SEQUENCE OVERLAP AUDIT ===")

for split_name, (X, y) in splits.items():
    total_rows = len(X)
    overlap_rows = 0
    same_label_rows = 0
    conflicting_label_rows = 0

    for sequence, label in zip(X, y):
        key = sequence.tobytes()
        other_labels = set()

        for other_split, labels in sequence_labels[key].items():
            if other_split != split_name:
                other_labels.update(labels)

        if other_labels:
            overlap_rows += 1

            if int(label) in other_labels:
                same_label_rows += 1

            if any(other_label != int(label)
                   for other_label in other_labels):
                conflicting_label_rows += 1

    percentage = 100 * overlap_rows / total_rows

    print(f"\n{split_name.upper()}")
    print(f"Total rows: {total_rows:,}")
    print(f"Rows with sequence overlap in another split: "
          f"{overlap_rows:,} ({percentage:.2f}%)")
    print(f"Overlapping rows with a matching label: "
          f"{same_label_rows:,}")
    print(f"Overlapping rows with a conflicting label: "
          f"{conflicting_label_rows:,}")

print("\nAudit complete. No model training or test-set tuning performed.")
