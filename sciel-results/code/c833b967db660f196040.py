import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the roots of the generalized Laguerre polynomial L_12^(-1/2)(x)
    as the double-root locations. These roots naturally align with the asymptotic
    zero distribution of Laguerre polynomials, minimizing the residual quotient's
    largest root B.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    # Compute the roots of L_12^(-1/2)(x)
    n = 12
    alpha = -0.5
    
    # scipy.special.roots_genlaguerre(n, alpha) returns the roots of L_n^(alpha)(x)
    x, w = roots_genlaguerre(n, alpha)
    
    # Convert to list of floats, sorted
    z_list = sorted([float(z) for z in x])
    
    # Validate: all positive, finite, distinct
    if len(z_list) != 12:
        # Fallback to Bessel-like spacing if something went wrong
        k_vals = np.arange(1, 13, dtype=np.float64)
        z_vals = (k_vals * np.pi) ** 2
        z_list = [float(z) for z in z_vals]
    
    # Final validation check
    if any(z <= 0 or not np.isfinite(z) for z in z_list):
        k_vals = np.arange(1, 13, dtype=np.float64)
        z_vals = (k_vals * np.pi) ** 2
        z_list = [float(z) for z in z_vals]
    
    if len(set(z_list)) != 12:
        k_vals = np.arange(1, 13, dtype=np.float64)
        z_vals = (k_vals * np.pi) ** 2
        z_list = [float(z) for z in z_vals]
    
    return z_list
