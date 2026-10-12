
from collections import defaultdict, Counter
import numpy as np
import preprocess

# Group all traces by their encoded, padded/truncated sequence.
groups = defaultdict(list)

for i, (sequence, label) in enumerate(zip(preprocess.X, preprocess.y)):
    key = tuple(sequence.tolist())
    groups[key].append({
        "index": i,
        "label": int(label),
        "length": int(preprocess.lengths[i]),
    })

conflicting_groups = {
    key: records
    for key, records in groups.items()
    if len({record["label"] for record in records}) > 1
}

print("\n========== CONFLICT INSPECTION ==========")
print("Conflicting unique encoded sequences:", len(conflicting_groups))

for number, (key, records) in enumerate(
    conflicting_groups.items(), start=1
):
    label_counts = Counter(record["label"] for record in records)

    print(f"\nConflict {number}")
    print("Label counts (0=Normal, 1=Anomaly):", dict(label_counts))
    print("Occurrences:", len(records))

    # Show a few original rows to investigate truncation or labeling.
    for record in records[:5]:
        i = record["index"]
        row = preprocess.df.iloc[i]

        print({
            "row_index": i,
            "BlockId": row.get("BlockId", "Unavailable"),
            "Label": row["Label"],
            "effective_length": record["length"],
            "original_features": str(row["Features"])[:250],
        })

print("\nConflict inspection completed.")
