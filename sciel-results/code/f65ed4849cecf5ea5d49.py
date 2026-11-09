"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    # Deadline guard: return best result before 600s limit
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Use the 12 positive roots of the generalized Laguerre polynomial L_12^(1)(x)
    # These are the extremal nodes for the weight function x * exp(-x)
    # which is theoretically optimal for this specific moment problem
    m = 12
    n = 1
    try:
        roots, _ = roots_genlaguerre(m, n)
        # roots_genlaguerre returns roots of L_m^(alpha)(x) for alpha=n
        # These should all be positive
        result = [float(r) for r in sorted(roots)]
        
        # Verify all roots are positive
        if all(r > 0 for r in result) and len(result) == 12:
            return result
    except Exception:
        pass
    
    # Fallback to seed if Laguerre roots fail
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
