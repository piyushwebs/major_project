
from collections import defaultdict
import preprocess

# Original dataset row indices reported by the error-analysis script.
error_rows = [
    171, 25159, 108042, 108049, 109457, 109756,
    83395, 213832, 108020, 108047, 114676, 286065,
    348183, 351940, 352805
]

# Count Normal and Anomaly labels for each complete sequence.
sequence_label_counts = defaultdict(lambda: [0, 0])

for sequence, label in zip(preprocess.sequences, preprocess.y):
    key = tuple(sequence)
    sequence_label_counts[key][int(label)] += 1

print("\n=== LABEL CONSISTENCY OF MISCLASSIFIED ROWS ===")

for row_idx in error_rows:
    sequence = tuple(preprocess.sequences[row_idx])
    actual_label = int(preprocess.y[row_idx])

    normal_count, anomaly_count = sequence_label_counts[sequence]
    conflicting = normal_count > 0 and anomaly_count > 0

    print(f"\nOriginal row: {row_idx}")
    print("Actual label:", "Anomaly" if actual_label else "Normal")
    print("Sequence length:", len(sequence))
    print("Occurrences labeled Normal:", normal_count)
    print("Occurrences labeled Anomaly:", anomaly_count)
    print("Conflicting labels:", conflicting)

print("\nAudit complete.")
