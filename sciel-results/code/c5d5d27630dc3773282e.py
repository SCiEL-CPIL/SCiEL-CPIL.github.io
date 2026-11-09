"""Symbolic regression using a hybrid RBF and Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using 5 Gaussian RBFs and 5 Chebyshev polynomial terms.
    
    Parameters (length 10):
    - params[0:5]: Amplitudes for 5 Gaussian RBFs with fixed centers and widths.
    - params[5:10]: Coefficients for Chebyshev polynomials T_0 to T_4.
    
    The RBF centers are fixed at -1.5, -0.75, 0.0, 0.75, 1.5.
    The RBF widths (sigma) are fixed at 0.6.
    The Chebyshev polynomials are computed on t = x/2 in [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    
    # --- RBF Part ---
    # Fixed centers and widths
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    sigma = 0.6
    
    # Compute RBF terms: exp(-((x - c) / sigma)^2)
    # Shape: (N_points, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_terms = np.exp(-(diff / sigma) ** 2)
    
    # RBF contribution: sum(params[0:5] * rbf_terms)
    rbf_result = np.dot(rbf_terms, params[0:5])
    
    # --- Chebyshev Part ---
    # Scale x to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Chebyshev contribution: sum(params[5:10] * T_k)
    cheb_result = (
        params[5] * T0
        + params[6] * T1
        + params[7] * T2
        + params[8] * T3
        + params[9] * T4
    )
    
    result = rbf_result + cheb_result
    
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
