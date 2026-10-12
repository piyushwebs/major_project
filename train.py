
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import precision_score, recall_score, f1_score

import preprocess
from dataset import create_dataloaders
from model import HDFSLSTM


# ==========================================
# Configuration
# ==========================================

SEED = 42
EPOCHS = 10
LEARNING_RATE = 0.001
CHECKPOINT_PATH = "best_hdfs_lstm_grouped.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# Group-aware Data Split and DataLoaders
# ==========================================

# Give identical complete event sequences the same group ID.
group_lookup = {}
groups = np.empty(len(preprocess.sequences), dtype=np.int64)

for i, sequence in enumerate(preprocess.sequences):
    key = tuple(sequence)

    if key not in group_lookup:
        group_lookup[key] = len(group_lookup)

    groups[i] = group_lookup[key]

print("Unique complete-sequence groups:", len(group_lookup))

splitter = StratifiedGroupKFold(
    n_splits=20,
    shuffle=True,
    random_state=SEED
)

folds = list(
    splitter.split(
        np.zeros(len(preprocess.y)),
        preprocess.y,
        groups
    )
)

# 14 folds train, 3 validation, 3 test.
train_idx = np.concatenate([folds[i][1] for i in range(14)])
val_idx = np.concatenate([folds[i][1] for i in range(14, 17)])
test_idx = np.concatenate([folds[i][1] for i in range(17, 20)])

assert len(set(groups[train_idx]) & set(groups[val_idx])) == 0
assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0
assert len(set(groups[val_idx]) & set(groups[test_idx])) == 0

print("Group overlap checks passed.")

# Select the existing preprocessed data using these indices.
X_train, lengths_train, y_train = (
    preprocess.X[train_idx],
    preprocess.lengths[train_idx],
    preprocess.y[train_idx]
)

X_val, lengths_val, y_val = (
    preprocess.X[val_idx],
    preprocess.lengths[val_idx],
    preprocess.y[val_idx]
)

X_test, lengths_test, y_test = (
    preprocess.X[test_idx],
    preprocess.lengths[test_idx],
    preprocess.y[test_idx]
)

print("\nGroup-aware split sizes:")
print(f"Train: {len(train_idx):,}")
print(f"Validation: {len(val_idx):,}")
print(f"Test: {len(test_idx):,}")

print("\nAnomaly counts:")
print("Train:", int((y_train == 1).sum()))
print("Validation:", int((y_val == 1).sum()))
print("Test:", int((y_test == 1).sum()))

train_loader, val_loader, test_loader = create_dataloaders(
    X_train, lengths_train, y_train,
    X_val, lengths_val, y_val,
    X_test, lengths_test, y_test
)

print("Training batches per epoch:", len(train_loader))



# ==========================================
# Model, loss and optimizer
# ==========================================

model = HDFSLSTM(
    vocab_size=30,
    embedding_dim=32,
    hidden_dim=64,
    num_layers=1
).to(device)

normal_count = int((y_train == 0).sum())
anomaly_count = int((y_train == 1).sum())

pos_weight = torch.tensor(
    [normal_count / anomaly_count],
    dtype=torch.float32,
    device=device
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Positive class weight: {pos_weight.item():.4f}")


# ==========================================
# Validation
# ==========================================

def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for X_batch, lengths_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            lengths_batch = lengths_batch.to(device)
            y_batch = y_batch.to(device)

            # Pass actual sequence lengths to the LSTM
            logits = model(X_batch, lengths_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()

            all_labels.extend(
                y_batch.cpu().numpy().astype(int)
            )
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = total_loss / len(loader.dataset)

    precision = precision_score(
        all_labels, all_predictions, zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions, zero_division=0
    )
    f1 = f1_score(
        all_labels, all_predictions, zero_division=0
    )

    return average_loss, precision, recall, f1


# ==========================================
# Training loop
# ==========================================

best_val_f1 = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0

    for X_batch, lengths_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        lengths_batch = lengths_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Pass actual sequence lengths to the LSTM
        logits = model(X_batch, lengths_batch)
        loss = criterion(logits, y_batch)

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_train_loss += loss.item() * X_batch.size(0)

    train_loss = total_train_loss / len(train_loader.dataset)

    val_loss, val_precision, val_recall, val_f1 = evaluate(
        model, val_loader
    )

    print(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Precision: {val_precision:.4f} | "
        f"Recall: {val_recall:.4f} | "
        f"F1: {val_f1:.4f}"
    )

    # Save the best model based on validation F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "max_len": preprocess.MAX_LEN,
                "vocab_size": 30,
                "embedding_dim": 32,
                "hidden_dim": 64,
                "num_layers": 1
            },
            CHECKPOINT_PATH
        )

        print(f"  Saved best checkpoint: {CHECKPOINT_PATH}")


print("\nTraining finished.")
print(f"Best validation F1: {best_val_f1:.4f}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
