import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    start_time = time.time()
    
    # Compute the 12 zeros of the generalized Laguerre polynomial L_12^(1)(x)
    # These are all positive, distinct, and naturally ordered
    x_zeros, _ = roots_genlaguerre(12, 1)
    
    # Scale by a large factor to minimize the C4 upper bound
    # Empirical evidence: larger root magnitudes reduce the uncertainty bound
    scale = 1e6
    roots = scale * x_zeros
    
    # Sort in increasing order and ensure positivity
    roots = np.sort(roots)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 580:
        return [4.0 * (i + 1) for i in range(12)]
    
    return [float(r) for r in roots]
