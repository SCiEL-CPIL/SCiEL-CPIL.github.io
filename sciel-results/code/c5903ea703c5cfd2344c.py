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
    # Initialize with a random binary sequence
    np.random.seed(42) # Fixed seed for reproducibility
    signs = np.random.choice([-1, 1], size=N).astype(np.int32)
    
    current_metric = _compute_metric(signs)
    
    # Local search: try flipping each sign
    # We iterate until no improvement is found
    max_iters = 50
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
                
    # Final check: ensure we are at a strong boundary (all 1s or all -1s)
    # If the search got stuck or ended at a non-optimal point due to noise,
    # we can force the best of the two global minima.
    # The theoretical minimum is achieved by constant sequences.
    # Let's compare the current result with all-1s and all--1s.
    
    signs_all_1 = np.ones(N, dtype=np.int32)
    signs_all_neg1 = -np.ones(N, dtype=np.int32)
    
    m_current = _compute_metric(signs)
    m_1 = _compute_metric(signs_all_1)
    m_neg1 = _compute_metric(signs_all_neg1)
    
    if m_1 < m_current:
        signs = signs_all_1
    elif m_neg1 < m_current:
        signs = signs_all_neg1
        
    return signs
# EVOLVE-BLOCK-END
