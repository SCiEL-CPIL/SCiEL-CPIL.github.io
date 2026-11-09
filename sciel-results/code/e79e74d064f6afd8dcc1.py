"""Symbolic regression with Chebyshev polynomial basis plus trig and Gaussian terms."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate unknown function using a well-conditioned basis:
    - Chebyshev polynomials T0..T6 evaluated at x/2 (bounded on [-2,2])
    - sin(x), cos(x) for periodic components
    - exp(-x^2/4) for Gaussian-like feature
    
    All basis functions are O(1) magnitude on [-2,2], helping BFGS converge.
    """
    p = params  # 10 parameters
    
    # Scaled input for Chebyshev: t = x/2, so t in [-1, 1]
    t = x * 0.5
    
    # Chebyshev polynomials T_n(t)
    T0 = np.ones_like(x)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    T6 = 32.0 * t**6 - 48.0 * t**4 + 18.0 * t**2 - 1.0
    
    # Trigonometric terms
    S = np.sin(x)
    C = np.cos(x)
    
    # Gaussian term
    G = np.exp(-x * x * 0.25)
    
    # Combine: p[0]*T0 + p[1]*T1 + ... + p[9]*G
    result = (p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3 +
              p[4] * T4 + p[5] * T5 + p[6] * T6 +
              p[7] * S + p[8] * C + p[9] * G)
    
    return result
