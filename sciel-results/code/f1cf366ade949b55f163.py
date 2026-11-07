# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Polynomial-Fourier approximator.
    
    p[0] -> 1 (bias)
    p[1] -> x
    p[2] -> x^2
    p[3] -> x^3
    p[4] -> sin(x)
    p[5] -> cos(x)
    p[6] -> sin(2x)
    p[7] -> cos(2x)
    p[8] -> sin(3x)
    p[9] -> cos(3x)
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    result = (p[0] * np.ones_like(x) + 
              p[1] * x + 
              p[2] * x**2 + 
              p[3] * x**3 +
              p[4] * np.sin(x) +
              p[5] * np.cos(x) +
              p[6] * np.sin(2.0 * x) +
              p[7] * np.cos(2.0 * x) +
              p[8] * np.sin(3.0 * x) +
              p[9] * np.cos(3.0 * x))
    
    return result
# EVOLVE-BLOCK-END
