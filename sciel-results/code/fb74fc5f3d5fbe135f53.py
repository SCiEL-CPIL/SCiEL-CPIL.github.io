"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis on [0.1, 6] with 10 parameters.
    params[0]: constant (DC)
    params[1:]: coefficients for sin/cos terms
    
    We use frequencies scaled so that k=1 corresponds to one full period over [0.1, 6].
    Domain length = 5.9, so fundamental frequency = 2*pi/5.9
    
    Basis:
      f0 = 1
      f1 = sin(2*pi*(x-0.1)/5.9)
      f2 = cos(2*pi*(x-0.1)/5.9)
      f3 = sin(4*pi*(x-0.1)/5.9)
      f4 = cos(4*pi*(x-0.1)/5.9)
      f5 = sin(6*pi*(x-0.1)/5.9)
      f6 = cos(6*pi*(x-0.1)/5.9)
      f7 = sin(8*pi*(x-0.1)/5.9)
      f8 = cos(8*pi*(x-0.1)/5.9)
      f9 = sin(10*pi*(x-0.1)/5.9)
    """
    # Scale x to [0, 1] over the domain [0.1, 6]
    x_min = 0.1
    x_max = 6.0
    domain_len = x_max - x_min
    x_norm = (x - x_min) / domain_len  # in [0, 1]
    
    # Build Fourier basis
    # 1 DC + 4 pairs of (sin, cos) = 1 + 8 = 9 terms, need 10
    # Let's use 1 DC + 4 sin + 4 cos + 1 more (sin of 5th harmonic)
    
    theta = 2.0 * np.pi * x_norm
    
    # Build basis matrix: (n_points, 10)
    n = len(x)
    basis = np.ones((n, 10))
    
    # Frequencies: 1, 2, 3, 4, 5
    # sin terms at freq 1,2,3,4,5 -> 5 sin terms
    # cos terms at freq 1,2,3,4 -> 4 cos terms
    # Total: 1 + 5 + 4 = 10
    
    basis[:, 1] = np.sin(1.0 * theta)
    basis[:, 2] = np.cos(1.0 * theta)
    basis[:, 3] = np.sin(2.0 * theta)
    basis[:, 4] = np.cos(2.0 * theta)
    basis[:, 5] = np.sin(3.0 * theta)
    basis[:, 6] = np.cos(3.0 * theta)
    basis[:, 7] = np.sin(4.0 * theta)
    basis[:, 8] = np.cos(4.0 * theta)
    basis[:, 9] = np.sin(5.0 * theta)
    
    return basis @ params
# EVOLVE-BLOCK-END
