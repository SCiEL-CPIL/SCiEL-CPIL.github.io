"""Symbolic regression with 1 input - Chebyshev + Gaussian RBF Hybrid."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Low-degree Chebyshev polynomials + Gaussian RBFs.
    
    This model is linear in the parameters, ensuring that BFGS converges to the
    global minimum of the least squares problem, avoiding local minima pitfalls
    of non-linear bases like Tanh networks.
    
    10 parameters:
      params[0:5]: Coefficients for Chebyshev polynomials T0..T4 (scaled x)
      params[5:10]: Amplitudes for 5 Gaussian RBFs centered at -2, -1, 0, 1, 2
    
    The Chebyshev part captures global smooth trends efficiently and stably.
    The RBF part provides local flexibility to capture non-polynomial features.
    """
    # Scale x to [-1, 1] for standard Chebyshev polynomials
    # Original domain is [-3, 3]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T0 to T4
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    # T4(t) = 8t^4 - 8t^2 + 1
    
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    
    poly_part = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3 + c4 * T4
    
    # Compute Gaussian RBFs
    # Fixed centers: -2, -1, 0, 1, 2
    # Fixed width sigma = 0.8 (chosen to provide good overlap and localization)
    sigma = 0.8
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Calculate distances
    # x shape (N,), centers shape (5,)
    # dist shape (N, 5)
    dist = np.abs(x[:, np.newaxis] - centers[np.newaxis, :])
    
    # Gaussian kernel
    # G shape (N, 5)
    G = np.exp(-(dist**2) / (2 * sigma**2))
    
    # Weights for RBFs
    w_rbf = params[5:10]
    
    # RBF part
    rbf_part = G @ w_rbf
    
    y = poly_part + rbf_part
    
    return y
# EVOLVE-BLOCK-END
