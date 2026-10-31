"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve

def _compute_metric(signs):
    """
    Computes the C3-prime metric: max(|autoconv|) / (sum(signs))^2.
    Returns inf if sum is 0.
    """
    s = signs.astype(np.float64)
    integral = np.sum(s)
    if abs(integral) < 1e-12:
        return np.inf
    
    # Autoconvolution
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    
    return max_ac / (integral * integral)

def construct():
    """
    Constructs the optimal signed step function.
    Uses a discrete gradient descent heuristic on binary signs {-1, 1}.
    """
    # Initialize with all 1s, which is a strong candidate for the minimum.
    # For all 1s: integral = N, autoconv is triangular with max N.
    # Metric = N / N^2 = 1/N.
    
    signs = np.ones(N, dtype=np.int32)
    
    current_metric = _compute_metric(signs)
    
    # Local search: try flipping each sign
    # We iterate until no improvement is found
    max_iters = 10
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        for i in range(N):
            # Flip sign
            original = signs[i]
            signs[i] = -original
            
            new_metric = _compute_metric(signs)
            
            if new_metric < current_metric:
                current_metric = new_metric
                improved = True
            else:
                # Revert
                signs[i] = original
                
    return signs
# EVOLVE-BLOCK-END
