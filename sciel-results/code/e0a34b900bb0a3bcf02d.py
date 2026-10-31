"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve


def _score(signs):
    """Compute the C3-prime value for a binary sign sequence."""
    s = signs.astype(np.int32)
    integral = np.sum(s)
    if integral == 0:
        return np.inf
    denom = integral ** 2
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return float(max_ac) / denom


def construct():
    # Start with a random binary sequence
    signs = np.random.choice([-1, 1], size=N)
    
    best_signs = signs.copy()
    best_score = _score(signs)
    
    # Greedy local search with random restarts
    max_iterations = 50000
    it = 0
    no_improve = 0
    threshold = 100
    
    while it < max_iterations and no_improve < threshold:
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = signs[idx]
        signs[idx] = -old_val
        
        new_score = _score(signs)
        
        if new_score < best_score:
            best_signs = signs.copy()
            best_score = new_score
            no_improve = 0
        else:
            # Revert if no improvement
            signs[idx] = old_val
            no_improve += 1
        
        it += 1
    
    return best_signs
# EVOLVE-BLOCK-END
