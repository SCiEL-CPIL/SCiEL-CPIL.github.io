"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of a diverse set of basis functions to approximate
    various target functions. Uses all 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Build basis functions (all safe for x in [-3, 3])
    # p0: constant
    # p1: x
    # p2: x^2
    # p3: x^3
    # p4: x^4
    # p5: sin(x)
    # p6: cos(x)
    # p7: exp(x) - clipped to avoid overflow, but x in [-3,3] is fine
    # p8: exp(-x)
    # p9: x^5
    
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4,
        np.sin(x),
        np.cos(x),
        np.exp(x),
        np.exp(-x),
        x**5,
    ])
    
    return basis @ params
# EVOLVE-BLOCK-END
