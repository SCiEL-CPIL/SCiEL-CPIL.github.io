"""Symbolic regression with a bounded, well-conditioned Gaussian basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a linear combination of 10
    bounded Gaussian basis functions with fixed centers and widths.
    The basis is well-conditioned and bounded, preventing BFGS divergence.
    """
    # Ensure x is 1-D
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Guard against non-finite params
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (10,) or not np.all(np.isfinite(p)):
        p = np.ones(10)
    
    # Fixed centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed bandwidth: sigma chosen so adjacent Gaussians overlap moderately
    # Spacing is 4/9 ≈ 0.444; use sigma = spacing/2 for good overlap
    sigma = (4.0 / 9.0) / 2.0
    
    # Build basis matrix: B[i, j] = exp(-0.5 * ((x[i] - centers[j]) / sigma)^2)
    # Vectorized computation
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    B = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Prediction
    y = B @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        y = np.zeros_like(x)
    
    return y
# EVOLVE-BLOCK-END
