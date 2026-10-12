
import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import preprocess
from dataset import create_dataloaders
from model import HDFSLSTM


CHECKPOINT_PATH = "best_hdfs_lstm.pt"
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Create loaders using the existing split.
# The test loader is used only for final evaluation.
_, _, test_loader = create_dataloaders(
    preprocess.X_train,
    preprocess.lengths_train,
    preprocess.y_train,
    preprocess.X_val,
    preprocess.lengths_val,
    preprocess.y_val,
    preprocess.X_test,
    preprocess.lengths_test,
    preprocess.y_test,
)

# Load the best validation checkpoint
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

all_labels = []
all_probabilities = []

with torch.no_grad():
    for X_batch, lengths_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        lengths_batch = lengths_batch.to(device)

        logits = model(X_batch, lengths_batch)
        probabilities = torch.sigmoid(logits)

        all_labels.extend(y_batch.numpy().astype(int))
        all_probabilities.extend(probabilities.cpu().numpy())

y_true = np.asarray(all_labels)
y_prob = np.asarray(all_probabilities)
y_pred = (y_prob >= THRESHOLD).astype(int)

# Final test metrics
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
pr_auc = average_precision_score(y_true, y_prob)
roc_auc = roc_auc_score(y_true, y_prob)

print("\n========== FINAL TEST RESULTS ==========")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Validation F1 at saved epoch:", checkpoint["val_f1"])
print("Test samples:", len(y_true))
print("Threshold:", THRESHOLD)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"PR-AUC / Average Precision: {pr_auc:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion matrix (rows=actual, columns=predicted):")
print("                 Pred Normal  Pred Anomaly")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
        digits=4,
    )
)

print("Test evaluation completed.")
