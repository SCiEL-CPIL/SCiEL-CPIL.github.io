"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    # Use the zeros of the Laguerre polynomial L_12(x) as optimal root locations
    # These are the natural nodes for the Laguerre quadrature/construction
    try:
        # roots_laguerre returns zeros and weights of L_n(x)
        zeros, weights = roots_laguerre(12)
        # The zeros are positive and sorted
        # Return as a list of 12 positive values
        result = [float(z) for z in zeros]
        # Ensure they're positive (Laguerre zeros are all positive for n>=1)
        if all(r > 0 for r in result):
            return result
        else:
            # Fallback: use seed values
            return [4.0 * (i + 1) for i in range(12)]
    except Exception:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
