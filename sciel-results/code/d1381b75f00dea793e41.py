import time
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    start_time = time.time()
    deadline = 600.0 - 30.0  # safety margin
    
    # Compute the 12 zeros of the generalized Laguerre polynomial L_12^(0)(x)
    # m=12, n=1 => try alpha=0 for standard exponential weight e^-x
    n_roots = 12
    alpha = 0
    
    # roots_genlaguerre returns zeros and weights for L_n^(alpha)(x)
    # Changing alpha changes the orthogonality weight, potentially aligning better with the C4 bound definition
    zeros, weights = roots_genlaguerre(n_roots, alpha)
    
    # Return the 12 positive double-root locations
    return [float(z) for z in zeros]
