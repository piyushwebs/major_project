
import torch
import preprocess
from dataset import create_dataloaders
from model import HDFSLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = create_dataloaders(
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

model = HDFSLSTM().to(device)
criterion = torch.nn.BCEWithLogitsLoss()

X, lengths, y = next(iter(train_loader))
X = X.to(device)
lengths = lengths.to(device)
y = y.to(device)

logits = model(X, lengths)
loss = criterion(logits, y)

assert logits.shape == y.shape
assert torch.isfinite(logits).all()
assert torch.isfinite(loss)

print("Device:", device)
print("Batch shape:", X.shape)
print("Lengths shape:", lengths.shape)
print("Labels shape:", y.shape)
print("Initial batch loss:", loss.item())
print("Training pipeline QA passed!")
