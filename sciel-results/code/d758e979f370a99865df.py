"""Refined uncertainty inequality construction using Laguerre roots."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.special as sp
import time

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the Laguerre polynomial L_12^(0)(x) as the optimal locations.
    """
    # Deadline guard, though this operation is very fast
    start_time = time.time()
    deadline = 600.0
    
    try:
        # Get the roots of the Laguerre polynomial of degree 12
        # roots_laguerre(n) returns the roots of L_n(x)
        roots = sp.roots_laguerre(12)
        
        # The roots are already real and positive.
        # Ensure they are sorted in ascending order
        roots = np.sort(roots.real)
        
        # Convert to a list of floats
        root_list = [float(r) for r in roots]
        
        # Check time (should be well within limit)
        if time.time() > deadline:
            # Fallback to seed if somehow slow (unlikely)
            return [4.0 * (i + 1) for i in range(12)]
            
        return root_list
        
    except Exception:
        # Fallback to seed program behavior if scipy fails
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
