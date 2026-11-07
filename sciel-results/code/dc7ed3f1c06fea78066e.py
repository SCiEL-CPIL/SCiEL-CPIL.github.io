"""Symbolic regression with RBF + polynomial basis."""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of inputs in [-3, 3]
    params: array of length 10
    Returns: array of predictions
    
    Basis:
      params[0] * 1
      params[1] * x
      params[2] * x^2
      params[3] * x^3
      params[4] * x^4
      params[5] * exp(-(x-(-3))^2 / (2*0.8^2))
      params[6] * exp(-(x-(-1.5))^2 / (2*0.8^2))
      params[7] * exp(-(x-0)^2 / (2*0.8^2))
      params[8] * exp(-(x-1.5)^2 / (2*0.8^2))
      params[9] * exp(-(x-3)^2 / (2*0.8^2))
    """
    # Polynomial terms
    y = params[0] + params[1] * x + params[2] * x**2 + params[3] * x**3 + params[4] * x**4
    
    # RBF terms with Gaussian kernels
    sigma = 0.8
    sigma2 = 2.0 * sigma * sigma
    
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Compute all RBF terms
    for i, c in enumerate(centers):
        y += params[5 + i] * np.exp(-(x - c)**2 / sigma2)
    
    return y
