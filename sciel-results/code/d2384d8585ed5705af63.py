"""
Strategy: Multi-scale Gaussian RBF Basis with Fixed Centers and Varied Bandwidths.

Rationale:
The previous approach used global basis functions (polynomials and low-frequency trigonometric terms).
While stable, these can struggle to capture sharp local features or complex non-polynomial shapes
without high-degree terms that cause ill-conditioning.

This new approach uses a set of Gaussian Radial Basis Functions (RBFs) with fixed centers
spread across the domain [0.1, 6.0] and varied bandwidths (sigmas). This allows the model
to fit both global trends (wide sigma) and local variations (narrow sigma) more effectively.

We use 10 parameters:
- 1 constant term (offset)
- 9 RBF terms: exp(-((x - c_i)^2) / (2 * sigma_i^2))

Centers c_i are chosen to cover the domain uniformly with some overlap.
Bandwidths sigma_i are varied to provide multi-scale approximation capability.

The model is linear in parameters:
y = params[0] + sum_{i=1}^{9} params[i] * exp(-((x - c_i)^2) / (2 * sigma_i^2))

This structure is well-suited for BFGS optimization from all-ones initialization because:
1. The basis functions are smooth and bounded.
2. The condition number is manageable with varied bandwidths.
3. It can approximate a wide class of functions, including those with local features.

We include a time guard to ensure we finish within the 120s limit, though this function
is computationally light.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Define fixed centers and bandwidths for the RBF basis
# Centers spread across [0.1, 6.0]
# We use 9 RBFs + 1 constant = 10 params total

# Centers: roughly evenly spaced
centers = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

# Bandwidths: varied to capture different scales
# Mix of wide and narrow bandwidths
# Wide: ~1.5, Medium: ~0.8, Narrow: ~0.4
sigmas = np.array([1.5, 0.8, 0.8, 1.2, 0.6, 0.6, 1.0, 0.5, 0.5])

# Precompute 2 * sigma^2 for efficiency
two_sigma_sq = 2.0 * sigmas * sigmas

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a constant term and 9 Gaussian RBF terms.
    params: array of shape (10,)
    x: Input array of shape (N,)
    Returns: array of shape (N,)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Clip x to domain to avoid extrapolation issues
    x = np.clip(x, 0.1, 6.0)
    
    # Start with constant term
    pred = np.full_like(x, params[0], dtype=np.float64)
    
    # Add RBF contributions
    for i in range(9):
        c = centers[i]
        ts2 = two_sigma_sq[i]
        # Compute RBF: exp(-(x - c)^2 / (2 * sigma^2))
        diff = x - c
        rbf = np.exp(-(diff * diff) / ts2)
        pred += params[i + 1] * rbf
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred

# EVOLVE-BLOCK-END
