"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import BSpline


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a piecewise cubic B-spline representation.
    The model is linear in the 10 control points (params), allowing for a robust
    least-squares fit via BFGS. The fixed knot vector ensures the basis functions
    cover the input domain [-2, 2] effectively.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define knot vector for cubic B-splines with 10 control points.
    # We need 10 + 3 + 1 = 14 knots.
    # We choose a uniform knot vector from -4 to 4 to ensure the active domain
    # [t_3, t_10] covers [-2, 2].
    # t_3 = -4 + 3 * (8/13) = -4 + 24/13 = -28/13 ~ -2.15
    # t_10 = -4 + 10 * (8/13) = -4 + 80/13 = 28/13 ~ 2.15
    t = np.linspace(-4.0, 4.0, 14)
    
    # Pre-compute the basis matrix B of shape (N, 10)
    # B[:, j] is the j-th cubic B-spline basis function evaluated at x
    n_points = len(x)
    B = np.zeros((n_points, 10))
    
    for j in range(10):
        # Coefficients for the j-th basis function: 1 at index j, 0 elsewhere
        c = np.zeros(10)
        c[j] = 1.0
        # Create BSpline object
        spl = BSpline(t, c, k=3)
        B[:, j] = spl(x)
        
    # The model is linear in params: y = B @ params
    return B @ params
# EVOLVE-BLOCK-END
