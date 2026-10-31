"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve

def _compute_max_ac(signs):
    """
    Computes the maximum absolute value of the autoconvolution.
    """
    s = signs.astype(np.float64)
    ac = fftconvolve(s, s, mode='full')
    return np.max(np.abs(ac))

def construct():
    """
    Constructs the optimal signed step function.
    Minimizes the maximum absolute autoconvolution.
    Assumes the denominator is effectively constant or large enough that minimizing the numerator is the primary goal, 
    consistent with the 'squared integral is constant' hint.
    """
    best_signs = None
    best_max_ac = np.inf
    
    # Multiple restarts to escape local minima
    num_restarts = 20
    
    for r in range(num_restarts):
        np.random.seed(42 + r)
        signs = np.random.choice([-1, 1], size=N).astype(np.int32)
        
        current_max_ac = _compute_max_ac(signs)
        
        # Local search: try flipping each sign
        # We iterate until no improvement is found
        max_iters = 100
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            best_flip_idx = -1
            best_flip_metric = current_max_ac
            
            for i in range(N):
                # Flip sign
                original = signs[i]
                signs[i] = -original
                
                new_max_ac = _compute_max_ac(signs)
                
                if new_max_ac < best_flip_metric:
                    best_flip_metric = new_max_ac
                    best_flip_idx = i
                    # Keep the flip for now to evaluate subsequent flips? 
                    # No, standard hill climbing evaluates one flip at a time from current state.
                    # But if we revert immediately, we might miss a path.
                    # Simple steepest descent: find the best single flip, apply it.
                
                # Revert to original for next evaluation
                signs[i] = original
            
            if best_flip_idx != -1:
                # Apply the best flip found
                signs[best_flip_idx] = -signs[best_flip_idx]
                current_max_ac = best_flip_metric
                improved = True
        
        # Check if this restart yielded a better global result
        if current_max_ac < best_max_ac:
            best_max_ac = current_max_ac
            best_signs = signs.copy()
            
    return best_signs
# EVOLVE-BLOCK-END
