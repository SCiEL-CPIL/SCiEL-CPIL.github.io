"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a truncated power series basis.
    This allows the model to fit polynomials up to degree 9,
    which can approximate a wide variety of smooth functions.
    
    params: array of length 10, representing coefficients c_0 ... c_9
    """
    # Ensure x is a 1D array for safety, though it should be
    x = np.asarray(x)
    
    # Build the Vandermonde-like matrix for powers 0 to 9
    # We want: c0*x^0 + c1*x^1 + ... + c9*x^9
    # Note: Standard Vandermonde is usually reversed, so we construct manually
    # or use np.vander and flip.
    
    # Using powers: x^0, x^1, ..., x^9
    # To avoid numerical instability with high powers of x in [-3, 3],
    # we can scale x or use orthogonal polynomials.
    # However, BFGS with initial ones might struggle if features are ill-conditioned.
    # Let's use a simple scaled polynomial basis or just standard powers.
    # Given x in [-3, 3], x^9 is up to 19683. This might cause conditioning issues.
    # A better approach: use a basis of scaled powers or Chebyshev polynomials.
    # Chebyshev polynomials are bounded in [-1, 1] on [-1, 1].
    # Let's map x from [-3, 3] to [-1, 1] for Chebyshev.
    # t = (x - 0) / 3  => t in [-1, 1]
    
    t = x / 3.0
    
    # Compute Chebyshev polynomials of the first kind T_0...T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n}(t) = 2t T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((len(x), 10))
    T[:, 0] = 1.0
    if len(x) > 0:
        T[:, 1] = t
        for n in range(2, 10):
            T[:, n] = 2.0 * t * T[:, n-1] - T[:, n-2]
    
    # The prediction is the dot product of T and params
    # T is (N, 10), params is (10,)
    pred = T @ params
    
    return pred
# EVOLVE-BLOCK-END
