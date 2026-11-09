"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    # The Cohn-Goncalves construction: 12 double roots for Laguerre combination (m=12, n=1)
    # The upper bound C4 is determined by the last sign change of the constructed function.
    # Using the zeros of the Laguerre polynomial L_12(x) as the root locations
    # is a natural choice since they are the optimal nodes for Laguerre quadrature
    # and minimize various extremal quantities.
    
    # Get the 12 zeros of the standard Laguerre polynomial L_12(x)
    try:
        roots, _ = roots_laguerre(12)
        # roots are already sorted in ascending order and all positive
        root_list = [float(r) for r in roots]
    except Exception:
        # Fallback to seed configuration
        root_list = [4.0 * (i + 1) for i in range(12)]
    
    # Verify: all roots must be positive
    if any(r <= 0 for r in root_list):
        root_list = [4.0 * (i + 1) for i in range(12)]
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Safety margin
        return root_list
    
    return root_list
# EVOLVE-BLOCK-END
