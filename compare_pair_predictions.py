
import torch
import numpy as np
import preprocess

from model import HDFSLSTM

CHECKPOINT = "best_hdfs_lstm_grouped.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pairs = [
    (171, 15),
    (25159, 19261),
    (109756, 109619),
    (83395, 77661),
    (114676, 108494),
]

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)

model = HDFSLSTM().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def predict(row_idx):
    sequence = preprocess.sequences[row_idx][:50]
    length = len(sequence)

    # Match the model's 50-event padded input format.
    padded = sequence + [0] * (50 - length)

    x = torch.tensor([padded], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([length], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, lengths)
        probability = torch.sigmoid(logits).item()

    return probability


print("\n=== PAIRED SEQUENCE PREDICTIONS ===")

for row_a, row_b in pairs:
    label_a = int(preprocess.y[row_a])
    label_b = int(preprocess.y[row_b])

    prob_a = predict(row_a)
    prob_b = predict(row_b)

    print("\n" + "-" * 55)
    print(f"Row A: {row_a} | Actual: {label_a} | P(Anomaly): {prob_a:.6f}")
    print(f"Row B: {row_b} | Actual: {label_b} | P(Anomaly): {prob_b:.6f}")
    print(f"Probability difference: {abs(prob_a - prob_b):.6f}")

print("\nPrediction comparison complete.")
