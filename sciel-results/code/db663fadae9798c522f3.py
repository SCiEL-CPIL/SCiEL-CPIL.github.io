"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Set a deadline guard. The evaluator kills after 150s.
# We will track time inside equation if it were a loop, but here it's vectorized.
# The main cost is in the optimizer, not in equation.
# However, we must ensure equation returns quickly.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a Legendre polynomial basis of degree 9.
    Legendre polynomials are orthogonal with respect to the uniform distribution
    on [-1, 1], which matches the data distribution. This improves the
    conditioning of the linear system for the BFGS optimizer compared to
    Chebyshev polynomials (which are orthogonal w.r.t. a different weight).
    """
    x_flat = np.asarray(x).flatten()
    n_points = x_flat.shape[0]
    
    if n_points == 0:
        return np.array([])
        
    # Map input x from [-3, 3] to [-1, 1]
    t = x_flat / 3.0
    
    # Precompute Legendre polynomials P_0 to P_9
    # P_0 = 1
    # P_1 = t
    # P_k = ((2k-1)/k * t * P_{k-1}) - ((k-1)/k * P_{k-2})
    
    P = np.zeros((10, n_points))
    P[0, :] = 1.0
    P[1, :] = t
    for k in range(2, 10):
        P[k, :] = ((2.0 * k - 1.0) / k) * t * P[k-1, :] - ((k - 1.0) / k) * P[k-2, :]
        
    # Linear combination: y = sum(params[i] * P[i, :])
    # params is shape (10,), P is shape (10, N)
    y = np.dot(params, P)
    
    return y
# EVOLVE-BLOCK-END
