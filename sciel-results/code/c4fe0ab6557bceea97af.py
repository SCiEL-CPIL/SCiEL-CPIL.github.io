"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_basis(x, y):
    """
    Build a well-conditioned basis of 10 features from (x, y).
    
    Uses a mix of:
    - Constant
    - Linear terms (x, y)
    - Interaction (xy)
    - Quadratic terms (x^2, y^2)
    - Cubic terms (x^3, y^3)
    - Higher-order interaction (x^2*y + x*y^2)
    
    This covers polynomials up to degree 3 plus one quartic interaction,
    giving flexibility for many smooth functions on [-1,1]^2.
    """
    n = len(x)
    basis = np.empty((n, 10), dtype=np.float64)
    
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = y
    basis[:, 3] = x2
    basis[:, 4] = y2
    basis[:, 5] = x * y
    basis[:, 6] = x3
    basis[:, 7] = y3
    basis[:, 8] = x2 * y + x * y2  # quartic interaction
    basis[:, 9] = x2 * y2           # quartic interaction
    
    return basis


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from (x, y) using a polynomial basis of 10 terms.
    
    The basis is designed to be well-conditioned on [-1,1]^2, allowing
    BFGS to converge reliably. The 10 parameters weight each basis function.
    """
    n = len(x)
    
    # Build the basis matrix
    B = _build_basis(x, y)
    
    # Prediction is B @ params
    pred = B @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback to simple linear if something went wrong
        pred = params[0] * x + params[1] * y + params[2]
    
    return pred
# EVOLVE-BLOCK-END
