
import numpy as np
import torch

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score,
)

import preprocess
from dataset import create_dataloaders
from model import HDFSLSTM


# ==========================================
# Configuration
# ==========================================

SEED = 42
CHECKPOINT_PATH = "best_hdfs_lstm_grouped.pt"
THRESHOLD = 0.5

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
    splitter.split(
        np.zeros(len(preprocess.y)),
        preprocess.y,
        groups,
    )
)

train_idx = np.concatenate([folds[i][1] for i in range(14)])
val_idx = np.concatenate([folds[i][1] for i in range(14, 17)])
test_idx = np.concatenate([folds[i][1] for i in range(17, 20)])

# Confirm complete-sequence groups do not cross splits.
assert len(set(groups[train_idx]) & set(groups[val_idx])) == 0
assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0
assert len(set(groups[val_idx]) & set(groups[test_idx])) == 0

print("Group overlap checks passed.")
print("Group-aware test rows:", len(test_idx))
print("Test anomalies:", int((preprocess.y[test_idx] == 1).sum()))


# ==========================================
# Create the held-out test DataLoader
# ==========================================

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
# Load the selected grouped checkpoint
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
print(f"Validation F1 at selection: {checkpoint['val_f1']:.4f}")


# ==========================================
# Evaluate ONCE on the held-out test set
# ==========================================

all_labels = []
all_probabilities = []

with torch.no_grad():
    for X_batch, lengths_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        lengths_batch = lengths_batch.to(device)

        logits = model(X_batch, lengths_batch)
        probabilities = torch.sigmoid(logits)

        all_labels.extend(y_batch.cpu().numpy().astype(int))
        all_probabilities.extend(probabilities.cpu().numpy())

y_true = np.asarray(all_labels)
y_prob = np.asarray(all_probabilities)
y_pred = (y_prob >= THRESHOLD).astype(int)


# ==========================================
# Report metrics
# ==========================================

print("\n=== GROUP-AWARE TEST RESULTS ===")
print("Test samples:", len(y_true))
print("Threshold:", THRESHOLD)

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1 score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
print(f"PR-AUC / Average Precision: {average_precision_score(y_true, y_prob):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")

print("\nConfusion matrix")
print("Rows = actual [Normal, Anomaly]")
print("Columns = predicted [Normal, Anomaly]")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        digits=4,
        zero_division=0,
    )
)

print("\nEvaluation complete.")
