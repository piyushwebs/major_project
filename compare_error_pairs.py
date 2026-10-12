import preprocess

pairs = [
    (171, 15),
    (25159, 19261),
    (109756, 109619),
    (83395, 77661),
    (114676, 108494),
]

def label_name(label):
    return "Anomaly" if int(label) == 1 else "Normal"

for row_a, row_b in pairs:
    seq_a = preprocess.sequences[row_a]
    seq_b = preprocess.sequences[row_b]

    print("\n" + "=" * 65)
    print(f"Row A: {row_a} | Label: {label_name(preprocess.y[row_a])}")
    print(f"Row B: {row_b} | Label: {label_name(preprocess.y[row_b])}")

    print("\nComplete sequence lengths:")
    print(f"A = {len(seq_a)}, B = {len(seq_b)}")

    print("\nFirst 50 events:")
    print("A:", seq_a[:50])
    print("B:", seq_b[:50])

    shared = min(len(seq_a), len(seq_b))
    differing_positions = [
        i for i in range(shared)
        if seq_a[i] != seq_b[i]
    ]

    print("\nDifferences within shared positions:")
    print("Number of differing positions:", len(differing_positions))
    print("Positions (zero-based):", differing_positions[:30])

    print("\nEvents after the first 50:")
    print("A:", seq_a[50:70])
    print("B:", seq_b[50:70])

print("\nPair comparison complete.")