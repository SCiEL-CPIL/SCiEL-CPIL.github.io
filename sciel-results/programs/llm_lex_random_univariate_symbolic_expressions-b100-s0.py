"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    # Precompute basis functions
    # 4 Chebyshev terms (global trend, well-conditioned)
    t = x / 3.0  # Map to [-1, 1]
    
    b0 = np.ones(n)
    b1 = t
    b2 = 2.0 * t**2 - 1.0
    b3 = 4.0 * t**3 - 3.0 * t
    
    # 6 Fourier terms (local oscillations)
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    b8 = np.sin(3.0 * x)
    b9 = np.cos(3.0 * x)
    
    # Stack into matrix (10, n)
    basis = np.vstack([
        b0, b1, b2, b3,
        b4, b5, b6, b7,
        b8, b9
    ])
    
    # Linear combination: (n,)
    result = basis.T @ params
    
    # Check deadline (though this is very fast)
    if time.time() > deadline:
        # Return best effort (which is already computed)
        pass
        
    return result
# EVOLVE-BLOCK-END
