"""Fourier series model for symbolic regression on [0.1, 6]."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier basis with 10 parameters:
      params[0]   : DC offset
      params[1]   : cos(w1 * x)
      params[2]   : sin(w1 * x)
      params[3]   : cos(w2 * x)
      params[4]   : sin(w2 * x)
      params[5]   : cos(w3 * x)
      params[6]   : sin(w3 * x)
      params[7]   : cos(w4 * x)
      params[8]   : sin(w4 * x)
      params[9]   : cos(w5 * x)
    
    Frequencies chosen to span the domain [0.1, 6] with good resolution.
    Using omega_k = k * pi / 6 for k=1..5 gives periods that naturally
    tile the domain, capturing both low and high frequency structure.
    """
    n = x.shape[0]
    # Frequencies: k * pi / 6 for k = 1, 2, 3, 4, 5
    w1 = np.pi / 6.0
    w2 = 2.0 * np.pi / 6.0
    w3 = 3.0 * np.pi / 6.0
    w4 = 4.0 * np.pi / 6.0
    w5 = 5.0 * np.pi / 6.0

    result = np.empty(n, dtype=np.float64)
    
    # DC term
    result += params[0]
    
    # Cosine and sine terms for each frequency
    result += params[1] * np.cos(w1 * x)
    result += params[2] * np.sin(w1 * x)
    result += params[3] * np.cos(w2 * x)
    result += params[4] * np.sin(w2 * x)
    result += params[5] * np.cos(w3 * x)
    result += params[6] * np.sin(w3 * x)
    result += params[7] * np.cos(w4 * x)
    result += params[8] * np.sin(w4 * x)
    result += params[9] * np.cos(w5 * x)
    
    return result
# EVOLVE-BLOCK-END
