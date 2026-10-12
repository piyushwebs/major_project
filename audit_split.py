
import numpy as np
import preprocess


def sequence_keys(X):
    # Event IDs are 0-29, so uint8 preserves every value.
    X_compact = np.asarray(X, dtype=np.uint8)
    return {row.tobytes() for row in X_compact}


train_keys = sequence_keys(preprocess.X_train)
val_keys = sequence_keys(preprocess.X_val)
test_keys = sequence_keys(preprocess.X_test)

print("\n========== SPLIT OVERLAP AUDIT ==========")

for name, keys in [
    ("Train vs Validation", val_keys),
    ("Train vs Test", test_keys),
    ("Validation vs Test", test_keys),
]:
    if name == "Train vs Validation":
        reference = train_keys
    elif name == "Train vs Test":
        reference = train_keys
    else:
        reference = val_keys

    overlap = reference & keys

    print(f"{name}:")
    print("  Unique sequences in reference:", len(reference))
    print("  Unique sequences in comparison:", len(keys))
    print("  Shared unique sequences:", len(overlap))

print("\n========== LABEL CONSISTENCY ==========")

# Check whether an identical encoded sequence has conflicting labels.
sequence_labels = {}
conflicts = 0

for X, y in [
    (preprocess.X_train, preprocess.y_train),
    (preprocess.X_val, preprocess.y_val),
    (preprocess.X_test, preprocess.y_test),
]:
    for row, label in zip(X, y):
        key = row.astype(np.uint8).tobytes()
        label = int(label)

        if key in sequence_labels and sequence_labels[key] != label:
            conflicts += 1
        else:
            sequence_labels[key] = label

print("Conflicting sequence-label observations:", conflicts)
print("Audit completed.")
