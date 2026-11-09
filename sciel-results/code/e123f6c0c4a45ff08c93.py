"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import BSpline

# Pre-compute knot vector for cubic spline (degree 3)
# We want 10 basis functions.
# Number of basis functions N = len(t) - (degree + 1)
# 10 = len(t) - 4 => len(t) = 14
# Knots: 4 at start, 4 at end, 6 interior
# Interior knots uniformly spaced in [-3, 3]
# -2.5, -1.5, -0.5, 0.5, 1.5, 2.5
_t_knots = np.array([-3, -3, -3, -3, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3, 3, 3, 3])
_degree = 3

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns prediction using a Cubic Spline basis with 6 interior knots.
    This is a linear model in the parameters, ensuring a convex loss landscape
    for BFGS, while providing local adaptivity superior to global polynomials.
    """
    start_time = time.time()
    
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.flatten()
        
    try:
        # Try using design_matrix for efficiency
        basis = BSpline.design_matrix(x, _t_knots, _degree)
    except AttributeError:
        # Fallback: evaluate each basis function individually
        n_points = x.size
        n_basis = 10
        basis = np.zeros((n_points, n_basis), dtype=np.float64)
        for i in range(n_basis):
            coeffs = np.zeros(n_basis)
            coeffs[i] = 1.0
            # Create spline with this basis function
            # Note: BSpline expects degree, not order
            s = BSpline(_t_knots, coeffs, _degree)
            basis[:, i] = s(x)
            
    pred = basis @ params
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 110:
         # If we are close to limit, we could try to return a simpler model?
         # But this is fast, so unlikely to trigger.
         pass
         
    return pred
# EVOLVE-BLOCK-END
