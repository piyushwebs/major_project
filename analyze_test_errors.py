
import numpy as np
import torch

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

import preprocess
from dataset import create_dataloaders
from model import HDFSLSTM


SEED = 42
CHECKPOINT_PATH = "best_hdfs_lstm_grouped.pt"
THRESHOLD = 0.5
MAX_EVENTS_TO_PRINT = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ==========================================
# Recreate the exact group-aware split
# ==========================================

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
    random_state=SEED,
)

folds = list(
    splitter.split(np.zeros(len(preprocess.y)), preprocess.y, groups)
)

train_idx = np.concatenate([folds[i][1] for i in range(14)])
val_idx = np.concatenate([folds[i][1] for i in range(14, 17)])
test_idx = np.concatenate([folds[i][1] for i in range(17, 20)])

assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0
assert len(set(groups[val_idx]) & set(groups[test_idx])) == 0

_, _, test_loader = create_dataloaders(
    preprocess.X[train_idx],
    preprocess.lengths[train_idx],
    preprocess.y[train_idx],
    preprocess.X[val_idx],
    preprocess.lengths[val_idx],
    preprocess.y[val_idx],
    preprocess.X[test_idx],
    preprocess.lengths[test_idx],
    preprocess.y[test_idx],
)


# ==========================================
# Load the selected checkpoint
# ==========================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

model = HDFSLSTM(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    num_layers=checkpoint["num_layers"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint:", CHECKPOINT_PATH)
print("Selected epoch:", checkpoint["epoch"])


# ==========================================
# Predict on test set without training
# ==========================================

all_probabilities = []

with torch.no_grad():
    for X_batch, lengths_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        lengths_batch = lengths_batch.to(device)

        logits = model(X_batch, lengths_batch)
        probabilities = torch.sigmoid(logits)

        all_probabilities.extend(probabilities.cpu().numpy())

y_true = preprocess.y[test_idx].astype(int)
y_prob = np.asarray(all_probabilities)
y_pred = (y_prob >= THRESHOLD).astype(int)

assert len(y_true) == len(y_prob) == len(test_idx)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\n=== ERROR SUMMARY ===")
print(f"True negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")
print("Total errors:", fp + fn)


# ==========================================
# Display misclassified sequences
# ==========================================

error_positions = np.flatnonzero(y_true != y_pred)

for position in error_positions:
    original_idx = int(test_idx[position])
    actual = int(y_true[position])
    predicted = int(y_pred[position])
    probability = float(y_prob[position])

    original_sequence = preprocess.sequences[original_idx]
    encoded_sequence = preprocess.X[original_idx]

    event_names = [f"E{event_id}" for event_id in original_sequence]
    shown_events = event_names[:MAX_EVENTS_TO_PRINT]

    if len(event_names) > MAX_EVENTS_TO_PRINT:
        shown_events.append("...")

    error_type = (
        "FALSE POSITIVE (normal flagged as anomaly)"
        if actual == 0
        else "FALSE NEGATIVE (anomaly missed)"
    )

    print("\n" + "=" * 65)
    print(error_type)
    print("Original dataset row:", original_idx)
    print("Actual label:", "Anomaly" if actual == 1 else "Normal")
    print("Predicted label:", "Anomaly" if predicted == 1 else "Normal")
    print(f"Predicted anomaly probability: {probability:.6f}")
    print("Original sequence length:", len(original_sequence))
    print("Events retained by model:", int((encoded_sequence != 0).sum()))
    print("Was sequence truncated:", len(original_sequence) > preprocess.MAX_LEN)
    print("Event sequence:", " ".join(shown_events))

print("\nError analysis complete.")
