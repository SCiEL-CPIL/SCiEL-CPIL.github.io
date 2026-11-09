"""Tensor-product Fourier basis for symbolic regression with 2 inputs."""
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a tensor-product Fourier basis.
    
    The model uses sine and cosine terms with distinct integer frequencies
    for both x and y, providing global periodicity and smooth interactions.
    
    Basis terms (10 total, matching 10 params):
      x-freq 1: cos(2πx), sin(2πx)
      y-freq 1: cos(2πy), sin(2πy)
      x-freq 2: cos(4πx), sin(4πx)
      y-freq 2: cos(4πy), sin(4πy)
      interaction: cos(2πx)cos(2πy)
      constant: 1
    
    This orthogonal basis ensures numerical stability for BFGS optimization
    and provides a complete representation for smooth periodic functions.
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Compute Fourier basis terms
    pi = np.pi
    two_pi = 2.0 * pi
    four_pi = 4.0 * pi
    
    # x-frequency 1
    cos_x1 = np.cos(two_pi * x)
    sin_x1 = np.sin(two_pi * x)
    
    # y-frequency 1
    cos_y1 = np.cos(two_pi * y)
    sin_y1 = np.sin(two_pi * y)
    
    # x-frequency 2
    cos_x2 = np.cos(four_pi * x)
    sin_x2 = np.sin(four_pi * x)
    
    # y-frequency 2
    cos_y2 = np.cos(four_pi * y)
    sin_y2 = np.sin(four_pi * y)
    
    # Interaction term
    interaction = cos_x1 * cos_y1
    
    # Build prediction as linear combination
    pred = (params[0] * cos_x1 +
            params[1] * sin_x1 +
            params[2] * cos_y1 +
            params[3] * sin_y1 +
            params[4] * cos_x2 +
            params[5] * sin_x2 +
            params[6] * cos_y2 +
            params[7] * sin_y2 +
            params[8] * interaction +
            params[9])
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
