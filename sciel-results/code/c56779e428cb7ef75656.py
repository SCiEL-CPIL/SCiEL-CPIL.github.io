"""Direct construction using exact zeros of the generalized Laguerre polynomial L_12^(1)(x)."""
import numpy as np
import time
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    """
    Return the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre construction with m=12, n=1. These are the exact zeros of
    the generalized Laguerre polynomial L_12^(1)(x), which have optimal
    orthogonality and stability properties for the Fourier sign uncertainty
    upper bound.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    # Get the exact zeros of L_12^(1)(x) using scipy
    # roots_genlaguerre(n, alpha) returns the roots of L_n^(alpha)(x)
    # Here n=12 (degree), alpha=1 (parameter for generalized Laguerre)
    roots, weights = roots_genlaguerre(12, 1)
    
    # The roots are already positive and sorted in ascending order
    # Verify they are all positive
    roots = np.asarray(roots, dtype=float)
    
    if len(roots) != 12 or np.any(roots <= 0):
        # Fallback: use a reasonable default
        roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    
    # Ensure sorted
    roots = np.sort(roots)
    
    elapsed = time.time() - start_time
    if elapsed + safety_margin > deadline:
        return [float(x) for x in roots]
    
    return [float(x) for x in roots]
# EVOLVE-BLOCK-END
