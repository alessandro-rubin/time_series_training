import torch.nn as nn

class SequenceToOneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):  # x: [B x T x D]
        _, (h_n, _) = self.lstm(x)   # h_n: [1 x B x H]
        out = self.fc(h_n.squeeze(0))  # [B x D]
        return out