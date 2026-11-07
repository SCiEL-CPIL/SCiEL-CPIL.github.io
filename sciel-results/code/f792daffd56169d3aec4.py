"""Rewritten program using a hybrid multi-scale Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid basis:
      - 1 constant basis function
      - 5 Gaussian RBFs at fixed centers with different widths
      - 4 polynomial basis functions (x, x^2, x^3, x^4)
    
    This hybrid allows localizing sharp features (via RBFs) while also
    capturing global trends (via polynomials). The 10 parameters are
    linearly fit by the evaluator using BFGS.
    
    y(x) = params[0] * 1
         + sum_{k=1}^{5} params[k] * exp(-((x - c_k)^2) / (2 * sigma_k^2))
         + params[5] * x + params[6] * x^2 + params[7] * x^3 + params[8] * x^4
         + params[9] * x^5
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    # Fixed centers for the 5 Gaussian RBFs, spread across [-2, 2]
    centers = np.array([-1.6, -0.8, 0.0, 0.8, 1.6])
    # Multi-scale widths: mix of narrow and wide to capture features at different scales
    sigmas = np.array([0.3, 0.5, 0.7, 1.0, 1.5])
    
    # Precompute the RBF contributions: shape (n, 5)
    # diff[i, k] = (x[i] - centers[k])
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 5)
    # sigma_sq: (5,) -> (1, 5) for broadcasting
    sigma_sq = (2.0 * sigmas) ** 2  # 2 * sigma^2 in denominator
    
    # Gaussian: exp(-diff^2 / (2 * sigma^2))
    rbf_terms = np.exp(-(diff ** 2) / sigma_sq[np.newaxis, :])  # (n, 5)
    
    # Polynomial terms: x, x^2, x^3, x^4, x^5
    poly_terms = np.column_stack([
        x,
        x ** 2,
        x ** 3,
        x ** 4,
        x ** 5
    ])  # (n, 5)
    
    # Build the full basis matrix: (n, 10)
    # Column 0: constant 1
    # Columns 1-5: RBF terms
    # Columns 6-10: polynomial terms
    basis = np.empty((n, 10), dtype=float)
    basis[:, 0] = 1.0
    basis[:, 1:6] = rbf_terms
    basis[:, 6:11] = poly_terms
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
