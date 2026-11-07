"""Symbolic regression with scaled polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute y = sum_{i=0}^{9} params[i] * (x/3)^i
    
    Uses scaled x to improve conditioning of the polynomial basis.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure we have at least 10 params
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), constant_values=0.0)
    
    # Scale x to [-1, 1] to improve numerical stability
    z = x / 3.0
    
    # Compute powers of z
    # b0 = 1
    # b1 = z
    # ...
    # b9 = z^9
    
    # Efficient computation using Horner's method or iterative multiplication
    # We need the linear combination: p[0]*1 + p[1]*z + ... + p[9]*z^9
    
    # Construct the result
    result = np.zeros_like(z)
    power = np.ones_like(z)
    
    for i in range(10):
        result += p[i] * power
        power *= z
        
    return result
# EVOLVE-BLOCK-END
