import torch.nn as nn
import torch.optim as optim
import torch
import mlflow
import mlflow.pytorch
import datetime as dt

# Train an autoencoder model on the provided training data
def train_autoencoder(model, X_train:torch.Tensor, epochs=50, lr=1e-3,run_name:None|str =None):
    model.train()  # set model to training mode
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if run_name is None:
        run_name = f"RUN_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "model": model.__class__.__name__,
            "epochs": epochs,
            "lr": lr,
            "input_dim": X_train.shape[-1],
            "window_size": X_train.shape[1],
        })
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
            avg_loss=total_loss/len(X_train)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        mlflow.pytorch.log_model(model, "model")