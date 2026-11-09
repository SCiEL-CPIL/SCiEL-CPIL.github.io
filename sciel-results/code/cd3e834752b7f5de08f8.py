"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_basis(x: np.ndarray, degree: int = 9) -> np.ndarray:
    """
    Compute Chebyshev polynomials of the first kind T_0(x) ... T_degree(x)
    evaluated at points x, mapped to [-1, 1] from [-3, 3].
    
    Returns shape (n_points, degree + 1).
    """
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    n = len(x)
    basis = np.zeros((n, degree + 1))
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = t
        for k in range(2, degree + 1):
            basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit a 9th-degree Chebyshev polynomial to the data using the 10 params.
    The evaluator uses BFGS to fit params; we just need to provide the
    forward pass.
    
    params[0..9] are the coefficients for T_0(x) ... T_9(x).
    """
    # Compute Chebyshev basis up to degree 9 (10 terms)
    basis = _chebyshev_basis(x, degree=9)
    
    # Linear combination
    return basis @ params[:10]
# EVOLVE-BLOCK-END
