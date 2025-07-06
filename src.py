import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def online_inference(model, X, scaler, window_size):
    model.eval()
    scores = []
    buffer = []

    with torch.no_grad():
        for x_t in X:
            buffer.append(x_t)
            if len(buffer) < window_size:
                scores.append(np.nan)
                continue
            window = np.stack(buffer[-window_size:])
            window_scaled = scaler.transform(window)
            x_in = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)
            x_rec = model(x_in).squeeze(0).numpy()
            score = np.mean((x_rec - window_scaled)**2)
            scores.append(score)
    return np.array(scores)


def train_autoencoder(model, X_train, epochs=50, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train):.4f}")


def generate_synthetic_data(T=1000, D=3, window=10):
    t = np.arange(T)
    X = np.stack([np.sin(0.01 * t * (d+1)) + 0.05 * np.random.randn(T) for d in range(D)], axis=1)
    X[300:305] += 3 * np.random.randn(5, D)
    X[700] += 6
    return X

def sliding_windows(X, window_size):
    return np.stack([X[i-window_size:i] for i in range(window_size, len(X))])


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        # Repeat hidden state for each timestep
        dec_input = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(dec_input)
        return out


class CNNAutoencoder(nn.Module):
    def __init__(self, input_dim, window_size):
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

    def forward(self, x):  # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # to (batch, features, seq_len)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)
