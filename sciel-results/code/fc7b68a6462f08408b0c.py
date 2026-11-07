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
    
    # Precompute basis functions - all O(1) on [-3,3] for good conditioning
    t = x / 3.0  # Map to [-1, 1]
    
    # Polynomial terms (normalized to be O(1))
    b0 = np.ones(n)
    b1 = t
    b2 = t**2
    b3 = t**3
    
    # Fourier terms at different frequencies
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    
    # Smooth nonlinear terms that capture local structure
    b8 = np.exp(-0.5 * x**2)  # Gaussian bump, max 1 at x=0
    b9 = x * np.exp(-0.5 * x**2)  # Odd Gaussian, captures asymmetric features
    
    # Stack into matrix (n, 10)
    basis = np.vstack([
        b0, b1, b2, b3,
        b4, b5, b6, b7,
        b8, b9
    ]).T
    
    # Linear combination: (n,)
    result = basis @ params
    
    # Check deadline (though this is very fast)
    if time.time() > deadline:
        # Return best effort (which is already computed)
        pass
        
    return result
# EVOLVE-BLOCK-END
