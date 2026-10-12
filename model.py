
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class HDFSLSTM(nn.Module):

    def __init__(
        self,
        vocab_size=30,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0
    ):
        super().__init__()

        # Convert event IDs into dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # LSTM processes only valid events
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Binary classification layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):

        # x: [batch_size, sequence_length]
        # lengths: [batch_size]

        # Convert event IDs into embeddings
        embedded = self.embedding(x)

        # Exclude padded timesteps from LSTM processing
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Process packed sequences
        _, (hidden, cell) = self.lstm(packed)

        # Final hidden state from the last LSTM layer
        last_hidden = hidden[-1]

        # Binary classification logits
        logits = self.fc(last_hidden)

        return logits.squeeze(1)
