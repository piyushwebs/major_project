
from collections import defaultdict, Counter
import preprocess

# Group rows by their final model input (after truncation and padding)
groups = defaultdict(list)

for i, (x, y) in enumerate(zip(preprocess.X, preprocess.y)):
    groups[tuple(x.tolist())].append(i)

print("\n========== FULL-SEQUENCE CONFLICT AUDIT ==========")

conflict_count = 0
truncation_collision_count = 0
identical_full_sequence_conflicts = 0

for encoded_sequence, indices in groups.items():
    labels = {int(preprocess.y[i]) for i in indices}

    if len(labels) < 2:
        continue

    conflict_count += 1

    # Recover each complete event sequence before truncation.
    full_groups = defaultdict(list)

    for i in indices:
        full_sequence = tuple(preprocess.sequences[i])
        full_groups[full_sequence].append(i)

    print(f"\nConflict {conflict_count}")
    print("Rows:", len(indices))
    print("Labels after preprocessing:", dict(
        Counter(int(preprocess.y[i]) for i in indices)
    ))
    print("Distinct complete sequences:", len(full_groups))

    if len(full_groups) > 1:
        truncation_collision_count += 1
        print("Finding: Different complete sequences share the same model input.")
    else:
        identical_full_sequence_conflicts += 1
        print("Finding: The complete event sequence has conflicting labels.")

    for full_sequence, row_indices in full_groups.items():
        full_labels = Counter(
            int(preprocess.y[i]) for i in row_indices
        )

        print(
            "  Full length:", len(full_sequence),
            "| Label counts:", dict(full_labels),
            "| Example BlockIds:",
            [
                preprocess.df.iloc[i].get("BlockId", "Unavailable")
                for i in row_indices[:3]
            ],
        )

print("\n========== SUMMARY ==========")
print("Conflicting encoded sequences:", conflict_count)
print("Groups with different complete sequences:", truncation_collision_count)
print("Groups with identical complete sequences but conflicting labels:",
      identical_full_sequence_conflicts)
