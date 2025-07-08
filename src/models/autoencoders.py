import torch
import torch.nn as nn

# LSTM-based autoencoder for sequence reconstruction
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)  # encode input sequence
        # Repeat hidden state for each timestep in the sequence
        dec_input = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(dec_input)  # decode to reconstruct sequence
        return out

# CNN-based autoencoder for sequence reconstruction
class CNNAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x:torch.Tensor):  # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # to (batch, features, seq_len) for Conv1d
        x = self.encoder(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)  # back to (batch, seq_len, features)


class FlattenedAutoencoder(nn.Module):
    def __init__(self, d, t, n1):
        super(FlattenedAutoencoder, self).__init__()
        self.d = d
        self.t = t
        self.n = d * t

        self.encoder = nn.Sequential(
            nn.Linear(self.n, n1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(n1, self.n),
            nn.Identity()  # or Identity if no normalization
        )

    def forward(self, x:torch.Tensor):
        # x: (batch_size, d, t)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, d * t)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), self.t, self.d)  # reshape back to (batch_size, d, t)
        return x
    

