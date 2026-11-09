"""Symbolic regression with 1 input - RBF + Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF + Chebyshev polynomial model.
    
    10 parameters:
      params[0]: bias (constant)
      params[1:5]: weights for 4 Gaussian RBFs centered at -2.25, -0.75, 0.75, 2.25
      params[5:10]: coefficients for Chebyshev polynomials T0..T4 (in normalized t)
    
    RBFs provide local flexibility (universal approximation), while Chebyshev
    polynomials capture global smooth structure with good conditioning.
    """
    # Normalize x from [-3,3] to [-1,1]
    t = x / 3.0
    
    # --- RBF components (Gaussian kernels) ---
    # Centers spread across the domain in original scale
    centers = np.array([-2.25, -0.75, 0.75, 2.25])
    sigma = 1.0  # fixed width; could be tuned but keep simple
    
    # Compute RBF values: exp(-((x - c)^2) / (2*sigma^2))
    # x shape: (n,), centers shape: (4,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 4)
    rbf_vals = np.exp(-0.5 * (diff / sigma) ** 2)  # (n, 4)
    
    # RBF contribution
    rbf_contrib = rbf_vals @ params[1:5]  # (n,)
    
    # --- Chebyshev polynomial components ---
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    # T4(t) = 8t^4 - 8t^2 + 1
    t0 = np.ones_like(t)
    t1 = t
    t2 = 2.0 * t**2 - 1.0
    t3 = 4.0 * t**3 - 3.0 * t
    t4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Chebyshev contribution
    cheb_contrib = (params[5] * t0 +
                    params[6] * t1 +
                    params[7] * t2 +
                    params[8] * t3 +
                    params[9] * t4)
    
    # Total prediction
    y = params[0] + rbf_contrib + cheb_contrib
    
    return y
# EVOLVE-BLOCK-END
