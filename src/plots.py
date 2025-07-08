import matplotlib.pyplot as plt
import numpy as np

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
