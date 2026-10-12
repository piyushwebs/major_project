
import numpy as np
import torch
import pandas as pd

import preprocess
from model import HDFSLSTM
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
CHECKPOINT = "best_hdfs_lstm_grouped.pt"
MAX_LEN = 50
BATCH_SIZE = 512
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Recreate the exact group-aware split used in training
# --------------------------------------------------
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
    random_state=42,
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

assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0
assert len(set(groups[val_idx]) & set(groups[test_idx])) == 0

# --------------------------------------------------
# Load the trained model
# --------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model = HDFSLSTM().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)
print("Selected epoch:", checkpoint.get("epoch", "not recorded"))
print("Group-aware test samples:", len(test_idx))

# --------------------------------------------------
# Predict test samples in batches
# --------------------------------------------------
X_test = preprocess.X[test_idx]
lengths_test = preprocess.lengths[test_idx]
y_test = preprocess.y[test_idx].astype(int)

probabilities = []

with torch.no_grad():
    for start in range(0, len(test_idx), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(test_idx))

        x = torch.tensor(
            X_test[start:end],
            dtype=torch.long,
            device=DEVICE,
        )

        lengths = torch.tensor(
            lengths_test[start:end],
            dtype=torch.long,
        )

        logits = model(x, lengths)
        probs = torch.sigmoid(logits).cpu().numpy()
        probabilities.extend(probs)

probabilities = np.asarray(probabilities)
predictions = (probabilities >= THRESHOLD).astype(int)

# --------------------------------------------------
# Overall metrics
# --------------------------------------------------
print("\n=== OVERALL TEST RESULTS ===")
print(f"Threshold: {THRESHOLD}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.6f}")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.6f}")
print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.6f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# Performance by original sequence length
# --------------------------------------------------
df = pd.DataFrame({
    "row": test_idx,
    "actual": y_test,
    "probability": probabilities,
    "prediction": predictions,
    "length": lengths_test,
})

df["result"] = np.where(
    df["actual"] == df["prediction"],
    "Correct",
    "Wrong",
)

df["length_group"] = pd.cut(
    df["length"],
    bins=[0, 10, 20, 30, 50, float("inf")],
    labels=["1-10", "11-20", "21-30", "31-50", "51+"],
    include_lowest=True,
)

print("\n=== PERFORMANCE BY SEQUENCE LENGTH ===")

for group_name, part in df.groupby("length_group", observed=False):
    if len(part) == 0:
        continue

    precision = precision_score(
        part["actual"],
        part["prediction"],
        zero_division=0,
    )
    recall = recall_score(
        part["actual"],
        part["prediction"],
        zero_division=0,
    )
    f1 = f1_score(
        part["actual"],
        part["prediction"],
        zero_division=0,
    )

    print(f"\nLength group: {group_name}")
    print("Samples:", len(part))
    print("Actual anomalies:", int(part["actual"].sum()))
    print("Errors:", int((part["result"] == "Wrong").sum()))
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1 score:  {f1:.6f}")

# --------------------------------------------------
# Probability distribution by actual label
# --------------------------------------------------
print("\n=== PROBABILITY DISTRIBUTION ===")

for label, name in [(0, "Normal"), (1, "Anomaly")]:
    scores = df.loc[df["actual"] == label, "probability"]

    print(f"\n{name} samples: {len(scores)}")
    print(f"Mean P(Anomaly):   {scores.mean():.6f}")
    print(f"Median P(Anomaly): {scores.median():.6f}")
    print(f"10th percentile:   {scores.quantile(0.10):.6f}")
    print(f"90th percentile:   {scores.quantile(0.90):.6f}")

# --------------------------------------------------
# False positives and false negatives
# --------------------------------------------------
false_positives = df[
    (df["actual"] == 0) & (df["prediction"] == 1)
].sort_values("probability", ascending=False)

false_negatives = df[
    (df["actual"] == 1) & (df["prediction"] == 0)
].sort_values("probability", ascending=True)

print("\n=== TOP FALSE POSITIVES ===")
print(
    false_positives[
        ["row", "length", "probability"]
    ].head(15).to_string(index=False)
)

print("\n=== TOP FALSE NEGATIVES ===")
print(
    false_negatives[
        ["row", "length", "probability"]
    ].head(15).to_string(index=False)
)

print("\nError distribution analysis complete.")
