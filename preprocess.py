
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "HDFS_v1" / "preprocessed" / "Event_traces.csv"
MAX_LEN = 50

# 0 is reserved for padding
EVENT_TO_ID = {f"E{i}": i for i in range(1, 30)}

# Load dataset
df = pd.read_csv(CSV_PATH)

required_columns = {"Features", "Label"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("Dataset loaded:", len(df), "traces")


# Convert event strings to integer sequences
def parse_features(feature_string):
    if pd.isna(feature_string):
        raise ValueError("Missing Features value found")

    feature_string = str(feature_string).strip()

    if feature_string.startswith("[") and feature_string.endswith("]"):
        feature_string = feature_string[1:-1]

    if not feature_string.strip():
        raise ValueError("Empty event sequence found")

    events = [event.strip() for event in feature_string.split(",")]

    unknown_events = [
        event for event in events if event not in EVENT_TO_ID
    ]

    if unknown_events:
        raise ValueError(f"Unknown event IDs: {unknown_events[:5]}")

    return [EVENT_TO_ID[event] for event in events]


sequences = []

for row_index, feature_string in enumerate(df["Features"]):
    try:
        sequences.append(parse_features(feature_string))
    except ValueError as error:
        raise ValueError(
            f"Invalid Features at CSV row {row_index + 2}: {error}"
        ) from error

# Record the effective sequence length AFTER truncation.
# This ensures it matches the sequence that the LSTM will receive.
lengths = np.array(
    [min(len(sequence), MAX_LEN) for sequence in sequences],
    dtype=np.int64
)

if np.any(lengths == 0):
    raise ValueError("Found a sequence with zero valid events")

# Pad sequences with 0; truncate sequences longer than MAX_LEN
X = np.zeros((len(sequences), MAX_LEN), dtype=np.int64)

for i, sequence in enumerate(sequences):
    sequence = sequence[:MAX_LEN]
    X[i, :len(sequence)] = sequence

# Map labels: Success = 0, Fail = 1
label_map = {"Success": 0, "Fail": 1}

y_series = df["Label"].map(label_map)

if y_series.isna().any():
    bad_labels = df.loc[y_series.isna(), "Label"].unique()
    raise ValueError(f"Unknown or missing labels: {bad_labels}")

y = y_series.to_numpy(dtype=np.int64)

# 70% train, 15% validation, 15% test
(
    X_train, X_temp,
    lengths_train, lengths_temp,
    y_train, y_temp
) = train_test_split(
    X,
    lengths,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

(
    X_val, X_test,
    lengths_val, lengths_test,
    y_val, y_test
) = train_test_split(
    X_temp,
    lengths_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Pipeline verification
print("\nPreprocessing complete")
print("MAX_LEN:", MAX_LEN)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nSplit shapes:")
print("Train:", X_train.shape, lengths_train.shape, y_train.shape)
print("Val:  ", X_val.shape, lengths_val.shape, y_val.shape)
print("Test: ", X_test.shape, lengths_test.shape, y_test.shape)

print("\nLabel counts:")
for name, labels in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    print(
        f"{name}: Normal={np.sum(labels == 0)}, "
        f"Anomaly={np.sum(labels == 1)}"
    )

# Confirm lengths match the number of non-padding events
assert np.all(lengths >= 1)
assert np.all(lengths <= MAX_LEN)
assert np.array_equal((X != 0).sum(axis=1), lengths)

# Confirm all event IDs and labels are valid
assert X.min() >= 0 and X.max() <= 29
assert set(np.unique(y)).issubset({0, 1})

print("\nAll preprocessing checks passed.")
