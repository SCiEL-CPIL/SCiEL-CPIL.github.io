"""
Strategy: High-Capacity Fourier Series on [-3, 3]
Instead of a low-capacity polynomial plus a few RBFs, we use a 10-term cosine series.
This provides a global, smooth basis that can represent a wide variety of functions.
The basis functions are cos(k * pi * x / 3) for k = 0..9.
The domain [-3, 3] maps naturally to the fundamental frequency.
The design matrix is well-conditioned (orthogonal-ish on the interval), 
which helps BFGS converge reliably from the all-ones start.
Parameters:
- p[0:10]: Coefficients for the 10 cosine basis functions.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a 10-term cosine series.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure we have at least 10 params
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
        
    # Extract coefficients
    coeffs = p[:10]
    
    # Construct basis matrix
    # k ranges from 0 to 9
    k = np.arange(10, dtype=np.float64)
    
    # Argument for cosine: k * pi * x / 3
    # x is (N,), k is (10,)
    # We want (N, 10) matrix
    arg = np.outer(x, k) * (np.pi / 3.0)
    
    # Compute cosines
    basis = np.cos(arg)
    
    # Linear combination
    y = basis @ coeffs
    
    return y

# EVOLVE-BLOCK-END
