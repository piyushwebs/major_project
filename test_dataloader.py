import torch
from preprocess import (
    X_train, lengths_train, y_train,
    X_val, lengths_val, y_val,
    X_test, lengths_test, y_test,
)
from dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    X_train, lengths_train, y_train,
    X_val, lengths_val, y_val,
    X_test, lengths_test, y_test,
)

X_batch, lengths_batch, y_batch = next(iter(train_loader))

print("X batch:", X_batch.shape, X_batch.dtype)
print("Lengths batch:", lengths_batch.shape, lengths_batch.dtype)
print("Labels batch:", y_batch.shape, y_batch.dtype)

assert X_batch.shape == (128, 50)
assert lengths_batch.shape == (128,)
assert y_batch.shape == (128,)

assert torch.all(lengths_batch >= 1)
assert torch.all(lengths_batch <= 50)
assert torch.equal(
    (X_batch != 0).sum(dim=1),
    lengths_batch
)

print("DataLoader QA passed!")
