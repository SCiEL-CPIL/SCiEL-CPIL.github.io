# EVOLVE-BLOCK-START
import numpy as np


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 probability distribution P[y, x] that maximizes the ratio
    H(X-Y) / max(H(X), H(X+Y), H(X+2Y)).
    
    This implementation uses a distribution uniform over the perimeter of the 10x10 grid.
    This shape tends to spread out the X-Y projection while keeping the X, X+Y, and X+2Y
    projections relatively concentrated compared to the uniform grid, potentially yielding
    a ratio > 1.
    """
    n = 10
    P = np.zeros((n, n), dtype=np.float64)
    
    # Mark perimeter cells
    for i in range(n):
        P[i, 0] = 1.0
        P[i, n-1] = 1.0
        P[0, i] = 1.0
        P[n-1, i] = 1.0
    
    # Normalize
    total = P.sum()
    if total > 0:
        P /= total
    else:
        # Fallback to uniform if something went wrong
        P = np.ones((n, n), dtype=np.float64) / (n * n)
        
    return P


# EVOLVE-BLOCK-END
