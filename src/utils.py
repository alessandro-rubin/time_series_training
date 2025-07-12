
import numpy as np

def make_seq_to_one_vectorized(X:np.ndarray, window_size):
    """
    X: array of shape (T, D)
    Returns:
        X_seq: shape (T - window_size, window_size, D)
        Y: shape (T - window_size, D)
    """
    T, D = X.shape
    if T <= window_size:
        raise ValueError("Time length must be greater than window_size.")
    
    X_seq = np.lib.stride_tricks.sliding_window_view(X, window_shape=(window_size,), axis=0)
    X_seq = X_seq[:-1]  # drop last window if needed to match Y
    Y :np.ndarray= X[window_size:]
    if not X_seq.shape[0] == Y.shape[0]:
        raise ValueError(f"Something went wrong, X_seq (shape {X_seq.shape}) and Y (shape {Y.shape}) don't have the same 0 shape.")
    return X_seq, Y