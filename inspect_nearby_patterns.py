from collections import defaultdict
import numpy as np
import preprocess

# Rows identified by analyze_test_errors.py
error_rows = [
    171, 25159, 108042, 108049, 109457, 109756,
    83395, 213832, 108020, 108047, 114676, 286065,
    348183, 351940, 352805
]

MAX_LEN = 50


def similarity(seq1, seq2):
    """Position-wise similarity over the shared sequence length."""
    if not seq1 or not seq2:
        return 0.0

    shared = min(len(seq1), len(seq2))
    matches = sum(seq1[i] == seq2[i] for i in range(shared))

    # Penalize differences in sequence length.
    return matches / max(len(seq1), len(seq2))


# Group complete sequences by label.
sequences_by_label = defaultdict(list)

for i, (sequence, label) in enumerate(
    zip(preprocess.sequences, preprocess.y)
):
    sequences_by_label[int(label)].append((i, sequence))

print("\n=== NEAREST OPPOSITE-LABEL PATTERNS ===")

for row_idx in error_rows:
    query = preprocess.sequences[row_idx]
    actual_label = int(preprocess.y[row_idx])
    opposite_label = 1 - actual_label

    query_first50 = query[:MAX_LEN]

    best_full = (0.0, None, None)
    best_first50 = (0.0, None, None)

    for candidate_idx, candidate in sequences_by_label[opposite_label]:
        if candidate_idx == row_idx:
            continue

        # Compare complete sequences.
        score_full = similarity(query, candidate)
        if score_full > best_full[0]:
            best_full = (score_full, candidate_idx, candidate)

        # Compare the first 50 events only.
        score_50 = similarity(
            query_first50,
            candidate[:MAX_LEN]
        )
        if score_50 > best_first50[0]:
            best_first50 = (score_50, candidate_idx, candidate)

    print(f"\nOriginal row: {row_idx}")
    print("Actual label:", "Anomaly" if actual_label else "Normal")
    print("Query length:", len(query))

    print("\nNearest opposite-label COMPLETE sequence:")
    print("Row:", best_full[1])
    print("Label:", "Anomaly" if opposite_label else "Normal")
    print("Similarity:", round(best_full[0], 4))
    print("Candidate length:", len(best_full[2]))

    print("\nNearest opposite-label FIRST-50 sequence:")
    print("Row:", best_first50[1])
    print("Similarity:", round(best_first50[0], 4))
    print("Candidate length:", len(best_first50[2]))

print("\nPattern inspection complete.")