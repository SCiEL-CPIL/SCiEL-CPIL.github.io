"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """Return 12 positive double-root locations using zeros of L_12^(1)(x)."""
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit
    
    # Get the 12 zeros of the generalized Laguerre polynomial L_12^(1)(x)
    # These are positive real numbers with optimal orthogonality properties
    zeros = roots_genlaguerre(12, 1)[0]
    
    # The zeros are already positive and sorted in increasing order
    # Verify they are all positive
    if np.any(zeros <= 0):
        # Fallback to seed if something goes wrong
        return [4.0 * (i + 1) for i in range(12)]
    
    # Return as a list of floats
    return [float(z) for z in zeros]
# EVOLVE-BLOCK-END
