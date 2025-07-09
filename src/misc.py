import numpy as np


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




def generate_synthetic_data_for_seq2one(T=1000, D=3,inject_anomaly=False):
    t = np.arange(T)
    X = np.stack([np.sin(0.01 * t * (d+1)) + 0.05 * np.random.randn(T) for d in range(D)], axis=1)
    y = np.zeros(T)
    if inject_anomaly:
        X[300:305] += 3 * np.random.randn(5, D)
        X[700] += 6
        y[300:305] = 1
        y[700] = 1
    return X, y

def make_sequence_to_one(X, window_size):
    X_in, Y_out = [], []
    for t in range(window_size, len(X)):
        X_in.append(X[t-window_size:t])
        Y_out.append(X[t])
    return np.array(X_in), np.array(Y_out)