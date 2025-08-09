import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceToOneLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):  # x: [B, T, D]
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))  # [B, D]

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])


class Predictor(nn.Module):
    def __init__(self, input_dim, n_timesteps, output_dim=1, fc_layers=[64, 32]):
        super().__init__()
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps
        #self.hidden_dim = hidden_dim
        #self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        layers = []
        self.flat_input_dim = input_dim * n_timesteps
        prev_dim = self.flat_input_dim
        for dim in fc_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor):
        # x: [batch_size, n_timesteps, input_dim]
        #_, (h_n, _) = self.lstm(x)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, d * t)
        out = self.fc(x)  # [batch_size, output_dim]
        return out
