
import torch
from torch.utils.data import Dataset, DataLoader


class HDFSDataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = torch.as_tensor(X, dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

        if not (
            len(self.X) == len(self.lengths) == len(self.y)
        ):
            raise ValueError("X, lengths, and y must have equal lengths")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.lengths[index], self.y[index]


BATCH_SIZE = 128


def create_dataloaders(
    X_train, lengths_train, y_train,
    X_val, lengths_val, y_val,
    X_test, lengths_test, y_test
):
    train_dataset = HDFSDataset(X_train, lengths_train, y_train)
    val_dataset = HDFSDataset(X_val, lengths_val, y_val)
    test_dataset = HDFSDataset(X_test, lengths_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, test_loader
