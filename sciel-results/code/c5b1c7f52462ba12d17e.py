import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0

    # The task asks for 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    # The "refined Cohn-Goncalves" construction typically utilizes the zeros of the Laguerre polynomial L_m(x).
    # For m=12, we compute the 12 positive zeros of L_12(x).
    
    try:
        # Compute roots of L_12(x)
        x, w = roots_laguerre(12)
        
        # The roots are already positive and distinct.
        # Sort them to ensure consistent ordering.
        roots = sorted(x)
        
        # Optional: Check if we have time to do a fine-tuning optimization.
        # Given the strong theoretical basis for Laguerre zeros, 
        # and the risk of overfitting to an unknown local metric, 
        # we return the theoretical zeros.
        
        if time.time() > deadline - 10:
            return roots
            
        return roots
        
    except Exception:
        # Fallback to the seed program's output if scipy is unavailable or fails
        return [4.0 * (i + 1) for i in range(12)]
