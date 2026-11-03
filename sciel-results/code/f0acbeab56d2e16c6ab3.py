"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the optimal sequence by solving the spectral optimization problem.
    
    The problem is to minimize max_t |ac[t]| / (sum h)^2.
    Let sum(h) = 1. We minimize max_t |ac[t]|.
    ac = h * h.
    In frequency domain, AC[k] = H[k]^2.
    ac[t] = IDFT(H[k]^2)[t].
    
    We know that max_t |ac[t]| >= ac[0] = sum(h[i]^2) >= 1/N (by Cauchy-Schwarz).
    The sequence h[i] = 1/N achieves exactly 1/N.
    This corresponds to H[0] = 1 and H[k] = 0 for k != 0.
    
    This program explicitly constructs this optimal spectral distribution.
    """
    # The optimal spectrum has all energy at DC (k=0)
    # H[0] = 1, H[k] = 0 for k != 0
    # This implies h is constant.
    h = np.ones(N) / N
    return h

# EVOLVE-BLOCK-END
