import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Fourier series with learnable angular frequencies and phases.
    Amplitudes are fixed to a deterministic 1/k decay sequence.
    
    params:
      0-4: angular frequencies w1..w5 (positive)
      5-9: phases phi1..phi5
    
    The model is:
      f(x) = sum_{k=1}^{5} (1/k) * cos(w_k * x + phi_k)
    
    This captures periodic and quasi-periodic structures through constructive
    interference of harmonics with optimally tuned frequencies and phases.
    The optimization landscape is smooth and well-conditioned for BFGS.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Extract frequencies and phases
    # Frequencies should be positive; use absolute value to ensure this
    w = np.abs(params[0:5])
    phi = params[5:10]
    
    # Fixed amplitude decay: 1/k for k = 1, 2, 3, 4, 5
    amplitudes = 1.0 / np.arange(1, 6, dtype=np.float64)
    
    # Compute the sum of cosines
    # Shape: (5, N) for broadcasting
    w_expanded = w[:, np.newaxis]  # (5, 1)
    phi_expanded = phi[:, np.newaxis]  # (5, 1)
    amp_expanded = amplitudes[:, np.newaxis]  # (5, 1)
    x_expanded = x[np.newaxis, :]  # (1, N)
    
    # Compute w_k * x + phi_k for all k and all x
    args = w_expanded * x_expanded + phi_expanded  # (5, N)
    
    # Compute cosines and weighted sum
    cosines = np.cos(args)  # (5, N)
    result = np.sum(amp_expanded * cosines, axis=0)  # (N,)
    
    return result
