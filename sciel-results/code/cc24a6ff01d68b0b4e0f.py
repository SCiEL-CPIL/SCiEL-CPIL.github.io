"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-basis linear model for symbolic regression with 1 input.
    
    Uses 10 basis functions to approximate the target function.
    The parameters are linear coefficients, so BFGS will converge to
    the global minimum of the training MSE (convex problem).
    
    Basis functions:
    1. 1 (constant)
    2. x
    3. x^2
    4. x^3
    5. sin(x)
    6. cos(x)
    7. exp(x/3)  (scaled to avoid overflow for x in [-3,3])
    8. tanh(x)
    9. |x|
    10. x * sin(x)
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure x is 1D for consistency
    if x.ndim > 1:
        x = x.ravel()
    
    # Compute basis functions
    basis = np.column_stack([
        np.ones_like(x),          # 1
        x,                        # 2
        x ** 2,                   # 3
        x ** 3,                   # 4
        np.sin(x),                # 5
        np.cos(x),                # 6
        np.exp(x / 3.0),          # 7 (scaled)
        np.tanh(x),               # 8
        np.abs(x),                # 9
        x * np.sin(x),            # 10
    ])
    
    # Linear combination: params @ basis.T
    return basis @ params
# EVOLVE-BLOCK-END
