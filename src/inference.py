import numpy as np
import torch


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
