"""Symbolic regression: recover an unknown function from 1 input with 10 params."""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis with constant, sine, and cosine terms at integer
    frequencies up to k=4 (9 terms), plus a single linear term x as the 10th
    parameter to capture non-periodic drift.
    
    The trigonometric functions are orthogonal on the symmetric domain [-3, 3],
    providing a well-conditioned basis for BFGS optimization.
    
    Parameters:
        x: input array of shape (N,)
        params: array of shape (10,)
            p[0]: constant
            p[1]: sin(x)
            p[2]: cos(x)
            p[3]: sin(2x)
            p[4]: cos(2x)
            p[5]: sin(3x)
            p[6]: cos(3x)
            p[7]: sin(4x)
            p[8]: cos(4x)
            p[9]: linear term x (captures non-periodic drift)
    """
    p = params
    
    # Compute Fourier basis terms
    term_const = np.ones_like(x)
    term_sin1 = np.sin(x)
    term_cos1 = np.cos(x)
    term_sin2 = np.sin(2 * x)
    term_cos2 = np.cos(2 * x)
    term_sin3 = np.sin(3 * x)
    term_cos3 = np.cos(3 * x)
    term_sin4 = np.sin(4 * x)
    term_cos4 = np.cos(4 * x)
    term_lin = x
    
    result = (p[0] * term_const +
              p[1] * term_sin1 +
              p[2] * term_cos1 +
              p[3] * term_sin2 +
              p[4] * term_cos2 +
              p[5] * term_sin3 +
              p[6] * term_cos3 +
              p[7] * term_sin4 +
              p[8] * term_cos4 +
              p[9] * term_lin)
    
    return result
