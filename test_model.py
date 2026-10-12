
import torch
from model import HDFSLSTM

# Use GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = HDFSLSTM().to(device)
model.eval()

# Simulate a batch of 128 sequences
X = torch.zeros(128, 50, dtype=torch.long)
lengths = torch.randint(1, 51, (128,))

# Fill each sequence with valid event IDs (1-29)
for i, length in enumerate(lengths):
    X[i, :length] = torch.randint(
        1, 30, (length.item(),)
    )

X = X.to(device)

with torch.no_grad():
    logits = model(X, lengths)

print("Device:", device)
print("Input shape:", X.shape)
print("Lengths shape:", lengths.shape)
print("Output shape:", logits.shape)
print("All outputs finite:", torch.isfinite(logits).all().item())

assert logits.shape == (128,)
assert torch.isfinite(logits).all()

print("Model QA passed!")
