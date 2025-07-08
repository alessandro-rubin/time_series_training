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




