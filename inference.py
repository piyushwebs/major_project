
from pathlib import Path

import torch

from model import HDFSLSTM

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MAX_LEN = 50
THRESHOLD = 0.5
VOCAB_SIZE = 30  # PAD=0, E1..E29 = 1..29

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "best_hdfs_lstm_grouped.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Load the already-trained model
# --------------------------------------------------
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = HDFSLSTM(vocab_size=VOCAB_SIZE).to(DEVICE)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from: {CHECKPOINT_PATH.name}")
    print(f"Device: {DEVICE}")

    return model


model = load_model()


# --------------------------------------------------
# Convert event tokens to trained vocabulary IDs
# --------------------------------------------------
def encode_events(events):
    if not events:
        raise ValueError("Event sequence cannot be empty.")

    encoded = []

    for event in events:
        token = str(event).strip().upper()

        if not token.startswith("E"):
            raise ValueError(
                f"Invalid event token: {event}. Expected E1 through E29."
            )

        try:
            event_id = int(token[1:])
        except ValueError:
            raise ValueError(f"Invalid event token: {event}")

        if not 1 <= event_id <= 29:
            raise ValueError(
                f"Unknown event: {event}. Expected E1 through E29."
            )

        encoded.append(event_id)

    # Match the training preprocessing: keep the first 50 events.
    encoded = encoded[:MAX_LEN]
    actual_length = len(encoded)

    # 0 is reserved for padding.
    padded = encoded + [0] * (MAX_LEN - actual_length)

    return padded, actual_length


# --------------------------------------------------
# Predict normal vs anomaly
# --------------------------------------------------
def predict_events(events, threshold=THRESHOLD):
    padded, actual_length = encode_events(events)

    x = torch.tensor(
        [padded],
        dtype=torch.long,
        device=DEVICE,
    )

    # pack_padded_sequence in model.py expects lengths on CPU.
    lengths = torch.tensor([actual_length], dtype=torch.long)

    with torch.inference_mode():
        logits = model(x, lengths)
        probability = torch.sigmoid(logits).item()

    predicted_label = (
        "Anomaly" if probability >= threshold else "Normal"
    )

    return {
        "predicted_label": predicted_label,
        "anomaly_probability": round(probability, 6),
        "threshold": threshold,
        "events_received": len(events),
        "events_used": actual_length,
    }
    


def predict_many_events(event_sequences, threshold=0.5, batch_size=128):
    """
    Predict multiple event sequences using the existing trained model.
    Each sequence must contain event IDs such as E5, E22, E11.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    if not event_sequences:
        return []

    # Use the same encoding, truncation and padding as single prediction.
    encoded_samples = [
        encode_events(events)
        for events in event_sequences
    ]

    results = []

    for start in range(0, len(encoded_samples), batch_size):
        batch = encoded_samples[start:start + batch_size]

        padded_sequences = [item[0] for item in batch]
        sequence_lengths = [item[1] for item in batch]

        x = torch.tensor(
            padded_sequences,
            dtype=torch.long,
            device=DEVICE,
        )

        lengths = torch.tensor(
            sequence_lengths,
            dtype=torch.long,
        )

        with torch.inference_mode():
            logits = model(x, lengths)
            probabilities = torch.sigmoid(logits).cpu().tolist()

        for offset, probability in enumerate(probabilities):
            index = start + offset
            actual_length = sequence_lengths[offset]

            label = (
                "Anomaly"
                if probability >= threshold
                else "Normal"
            )

            results.append({
                "predicted_label": label,
                "anomaly_probability": round(
                    float(probability), 6
                ),
                "threshold": threshold,
                "events_received": len(event_sequences[index]),
                "events_used": actual_length,
            })

    return results



# --------------------------------------------------
# Test the inference function
# --------------------------------------------------
if __name__ == "__main__":
    sample_events = [
        "E5", "E22", "E11", "E9", "E11", "E9", "E26"
    ]

    result = predict_events(sample_events)

    print("\n=== INFERENCE RESULT ===")
    for key, value in result.items():
        print(f"{key}: {value}")
