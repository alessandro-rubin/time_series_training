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



