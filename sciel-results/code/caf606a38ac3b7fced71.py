# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Refined Hybrid Polynomial-Fourier approximator.
    
    p[0] -> 1 (bias)
    p[1] -> x/3
    p[2] -> (x/3)^2
    p[3] -> (x/3)^3
    p[4] -> sin(x)
    p[5] -> cos(x)
    p[6] -> sin(2x)
    p[7] -> cos(2x)
    p[8] -> sin(4x)
    p[9] -> cos(4x)
    
    Polynomial terms scaled to O(1) magnitude on [-3,3] for better BFGS conditioning.
    Higher-frequency trig terms (4x) capture finer oscillatory structure.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Scaled polynomial basis
    x1 = x / 3.0
    x2 = x1 * x1
    x3 = x2 * x1
    
    # Trigonometric basis
    s1 = np.sin(x)
    c1 = np.cos(x)
    s2 = np.sin(2.0 * x)
    c2 = np.cos(2.0 * x)
    s4 = np.sin(4.0 * x)
    c4 = np.cos(4.0 * x)
    
    result = (p[0] * np.ones_like(x) + 
              p[1] * x1 + 
              p[2] * x2 + 
              p[3] * x3 +
              p[4] * s1 +
              p[5] * c1 +
              p[6] * s2 +
              p[7] * c2 +
              p[8] * s4 +
              p[9] * c4)
    
    return result
# EVOLVE-BLOCK-END
