"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-term Fourier series with fixed frequencies {1, 2, 3, 4, 5}.
    Basis: cos(kx), sin(kx) for k=1..5.
    This basis is orthogonal on [-3, 3] (approximately, for integer frequencies
    over symmetric intervals), leading to a well-conditioned linear system
    that BFGS can solve efficiently. It captures periodic and smooth components.
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure x is 1D for safety
    if x.ndim != 1:
        x = x.flatten()
        
    # Pre-compute basis functions
    # Frequencies: 1, 2, 3, 4, 5
    # Order of params: [c1, s1, c2, s2, c3, s3, c4, s4, c5, s5]
    
    result = np.zeros_like(x)
    
    for k in range(1, 6):
        arg = k * x
        c_k = np.cos(arg)
        s_k = np.sin(arg)
        
        # params index: 2*(k-1) for cos, 2*(k-1)+1 for sin
        idx_c = 2 * (k - 1)
        idx_s = 2 * (k - 1) + 1
        
        result += params[idx_c] * c_k + params[idx_s] * s_k
        
    return result
# EVOLVE-BLOCK-END
