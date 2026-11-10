"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev polynomial basis of degree 9.
    This uses all 10 parameters as coefficients for T_0(x) to T_9(x).
    The input x is scaled to [-1, 1] to improve numerical stability and conditioning.
    """
    # Scale x from [-3, 3] to [-1, 1]
    u = x / 3.0
    
    # Compute Chebyshev polynomials T_0(u) to T_9(u)
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    n_points = len(x)
    features = np.zeros((n_points, 10), dtype=np.float64)
    
    # T_0
    features[:, 0] = 1.0
    
    # T_1
    if n_points > 0:
        features[:, 1] = u
        
        # T_2 to T_9
        for k in range(2, 10):
            features[:, k] = 2.0 * u * features[:, k-1] - features[:, k-2]
            
    # Linear combination of features
    return features @ params
# EVOLVE-BLOCK-END
