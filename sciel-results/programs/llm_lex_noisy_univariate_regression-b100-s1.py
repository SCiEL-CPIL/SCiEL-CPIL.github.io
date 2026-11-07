"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Scaled Chebyshev Basis with RBF Refinement
#
# The previous strategy used a standard polynomial basis (x^0..x^4) on the raw
# domain [-3, 3]. This leads to an ill-conditioned Vandermonde matrix, making
# the gradient landscape for BFGS difficult to navigate, especially for higher
# degree terms.
#
# This strategy improves the basis in two ways:
# 1. **Domain Scaling**: We scale the input x from [-3, 3] to t in [-1, 1].
#    This normalizes the magnitude of the basis functions.
# 2. **Chebyshev Polynomials**: We replace the standard power basis with
#    Chebyshev polynomials T_k(t). Chebyshev polynomials are orthogonal on
#    [-1, 1] and have significantly better conditioning properties, leading to
#    more stable and accurate convergence for the linear least-squares subproblem
#    embedded in the BFGS optimization.
# 3. **Hybrid Basis**: We use 6 Chebyshev coefficients (T_0 to T_5) to capture
#    the global smooth trend, and 4 RBF amplitudes for local deviations.
#    This keeps the model linear in parameters, ensuring BFGS can reliably find
#    the optimal weights for this richer, better-conditioned basis.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Scaled Chebyshev + RBF Model.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            params[0:6] = Chebyshev coefficients for T_0(t) to T_5(t)
            params[6:10] = Amplitudes for 4 fixed RBFs
            
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # 1. Scale x to [-1, 1]
    # x is in [-3, 3], so t = x / 3.0
    t = x / 3.0
    
    # 2. Compute Chebyshev Polynomials T_0 to T_5
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    
    # Combine Chebyshev part
    # p[0] * T0 + ... + p[5] * T5
    cheb_part = (p[0] * T0 + 
                 p[1] * T1 + 
                 p[2] * T2 + 
                 p[3] * T3 + 
                 p[4] * T4 + 
                 p[5] * T5)
                 
    # 3. RBF Part
    # 4 RBFs to utilize the remaining 4 params
    N_RBF = 4
    # Centers: Symmetrically placed in [-3, 3]
    # Avoiding the exact boundaries to ensure coverage
    centers = np.array([-2.25, -0.75, 0.75, 2.25])
    
    # Fixed width
    sigma = 1.0
    sigma_sq_2 = 2.0 * sigma**2
    
    # Compute RBF values
    # x shape: (N,)
    # centers shape: (4,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff**2) / sigma_sq_2)
    
    # rbf_values shape: (N, 4)
    # p[6:10] shape: (4,)
    rbf_part = rbf_values @ p[6:10]
    
    result = cheb_part + rbf_part
    
    return result

# EVOLVE-BLOCK-END
