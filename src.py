import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Perform online inference with a trained autoencoder model on time series data
def online_inference(model, X, scaler, window_size):
    model.eval()  # set model to evaluation mode
    scores = []
    buffer = []

    with torch.no_grad():  # disable gradient computation
        for x_t in X:
            buffer.append(x_t)
            if len(buffer) < window_size:
                scores.append(np.nan)  # not enough data for a window yet
                continue
            window = np.stack(buffer[-window_size:])  # get latest window
            window_scaled = scaler.transform(window)  # scale window
            x_in = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)  # add batch dimension
            x_rec = model(x_in).squeeze(0).numpy()  # reconstruct window
            score = np.mean((x_rec - window_scaled)**2)  # compute reconstruction error
            scores.append(score)
    return np.array(scores)

# Train an autoencoder model on the provided training data
def train_autoencoder(model, X_train:torch.Tensor, epochs=50, lr=1e-3):
    model.train()  # set model to training mode
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        # create batches from training data
        for batch in torch.utils.data.DataLoader(X_train, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            recon = model(batch)  # forward pass
            loss = loss_fn(recon, batch)  # compute reconstruction loss
            loss.backward()  # backpropagation
            optimizer.step()  # update weights
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train):.4f}")

# Generate synthetic multivariate time series data with injected anomalies
def generate_synthetic_data(T=1000, D=3):
    t = np.arange(T)
    # Create D sinusoidal signals with noise
    X = np.stack([np.sin(0.01 * t * (d+1)) + 0.05 * np.random.randn(T) for d in range(D)], axis=1)
    X[300:305] += 3 * np.random.randn(5, D)  # inject anomaly
    X[700] += 6  # inject spike anomaly
    return X

# Create sliding windows from time series data
def sliding_windows(X, window_size):
    return np.stack([X[i-window_size:i] for i in range(window_size, len(X))])

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
            nn.Sigmoid()  # or Identity if no normalization
        )

    def forward(self, x:torch.Tensor):
        # x: (batch_size, d, t)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, d * t)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), self.t, self.d)  # reshape back to (batch_size, d, t)
        return x


import matplotlib.pyplot as plt

def plot_reconstruction(x_original, x_recon, signal_names=None, window_start=0):
    """
    x_original, x_recon: numpy arrays of shape [T x D]
    signal_names: optional list of signal names
    window_start: index of the window start (for x-axis labels)
    """
    T, D = x_original.shape
    time = np.arange(window_start, window_start + T)

    if signal_names is None:
        signal_names = [f'Signal {i}' for i in range(D)]

    fig, axs = plt.subplots(D, 1, figsize=(10, 2 * D), sharex=True)
    if D == 1:
        axs = [axs]

    for d in range(D):
        axs[d].plot(time, x_original[:, d], label='Actual', linewidth=1.5)
        axs[d].plot(time, x_recon[:, d], label='Reconstructed', linestyle='--')
        axs[d].set_ylabel(signal_names[d])
        axs[d].legend(loc='upper right')
        axs[d].grid(True)

    axs[-1].set_xlabel("Time")
    plt.suptitle("Actual vs Reconstructed Signals")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
