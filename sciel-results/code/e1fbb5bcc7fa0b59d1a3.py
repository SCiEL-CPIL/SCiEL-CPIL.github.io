"""
Symbolic regression using a linear Fourier-sine/cosine basis on [-2, 2].
This is a fundamentally different representation from polynomials or sigmoids.
The basis is linear in the 10 parameters, so BFGS (starting from ones) should
reliably find the global least-squares solution.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y(x) = p0 + sum_{k=1}^{4} [ p_{2k-1} * sin(k*pi*x/2) + p_{2k} * cos(k*pi*x/2) ]
    
    Using the first 4 harmonics (k=1..4) gives 9 frequency terms plus a constant = 10 params.
    This basis can represent a rich variety of smooth functions on [-2, 2].
    The linear structure ensures convex optimization and reliable fitting.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.ndim != 1:
        p = p.flatten()
    
    # Compute basis functions
    # p[0] is the constant term
    result = p[0] * np.ones_like(x)
    
    # For k = 1, 2, 3, 4:
    # sin term uses p[2*k-1], cos term uses p[2*k]
    for k in range(1, 5):
        arg = k * np.pi * x / 2.0
        result += p[2*k - 1] * np.sin(arg)
        result += p[2*k] * np.cos(arg)
    
    return result
# EVOLVE-BLOCK-END
