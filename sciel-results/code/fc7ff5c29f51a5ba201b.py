"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    try:
        x, w = roots_laguerre(12)
        x = np.sort(x)
        if len(x) == 12 and np.all(x > 0):
            return [float(xi) for xi in x]
    except Exception:
        pass
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
