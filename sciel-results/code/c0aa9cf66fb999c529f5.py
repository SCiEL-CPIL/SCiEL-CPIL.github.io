"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Chebyshev polynomial basis (degrees 0-9) on scaled domain [-1, 1].
    Chebyshev polynomials are orthogonal on [-1, 1] with respect to the weight 1/sqrt(1-u^2),
    which provides much better conditioning than monomials or Gaussians for BFGS optimization.
    With 10 parameters, we can represent a degree-9 Chebyshev polynomial, which is a
    universal approximator for continuous functions on [-1, 1].
    
    The key insight from the inspirations:
    - Inspiration 1 (Chebyshev): Good conditioning, but scored poorly possibly due to
      the specific implementation or the fact that BFGS from all-ones might get stuck.
    - Current program (Gaussians): Localized basis, but with fixed centers/widths,
      it may not capture global trends well and the condition number can be poor.
    
    By using Chebyshev polynomials on the scaled domain, we get:
    1. Orthogonality -> well-conditioned design matrix
    2. Global support -> can capture both local and global features
    3. Degree 9 -> sufficient capacity for smooth functions on [-3, 3]
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x from [-3, 3] to [-1, 1]
    u = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    T0 = np.ones_like(u)
    T1 = u.copy()
    
    # Build the prediction as a linear combination
    # result = sum_{k=0}^{9} p[k] * T_k(u)
    
    result = params[0] * T0 + params[1] * T1
    
    if len(params) > 2:
        T_prev2 = T0
        T_prev1 = T1
        for k in range(2, 10):
            T_curr = 2.0 * u * T_prev1 - T_prev2
            result += params[k] * T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr
    
    return result
# EVOLVE-BLOCK-END
