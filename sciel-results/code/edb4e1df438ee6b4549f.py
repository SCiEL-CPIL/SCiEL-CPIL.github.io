"""Symbolic regression with 1 input: Fourier series basis for periodic structure."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Fourier series basis.
    
    The model is: y = p_0 + sum_{k=1}^{5} [p_{2k-1} * sin(k*pi*x/3) + p_{2k} * cos(k*pi*x/3)]
    
    This explicitly captures periodic/oscillatory structure in the target function
    while maintaining a linear-in-params structure for robust BFGS convergence.
    The frequencies are scaled to the domain [-3, 3] so that k=1 corresponds to
    one full period over the input range.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Scale x to [-1, 1] for numerical stability
    t = x / 3.0
    
    # Build basis matrix (n_samples, 10)
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Constant term
    basis[:, 0] = 1.0
    
    # Fourier terms: sin(k*pi*t) and cos(k*pi*t) for k=1..5
    # Note: t is in [-1, 1], so k*pi*t covers k half-periods in the scaled domain
    # which corresponds to k full periods in the original [-3, 3] domain
    for k in range(1, 6):
        angle = k * np.pi * t
        basis[:, 2*k - 1] = np.sin(angle)
        basis[:, 2*k] = np.cos(angle)
    
    # Compute prediction: dot product of basis with params
    pred = basis @ params
    
    # Safe computation: check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
