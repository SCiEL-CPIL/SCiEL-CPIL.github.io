# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Piecewise linear "hat" basis with 10 parameters
    # 9 hat functions centered at uniformly spaced knots + 1 intercept
    
    n_knots = 9
    x_min, x_max = -2.0, 2.0
    h = (x_max - x_min) / n_knots
    
    # Knot positions
    knots = np.linspace(x_min, x_max, n_knots)
    
    result = np.zeros_like(x)
    
    # Compute hat functions
    # phi_0: left edge
    result += params[0] * np.maximum(0.0, 1.0 - (x - knots[0]) / h)
    
    # phi_i: interior knots
    for i in range(1, n_knots - 1):
        result += params[i] * np.maximum(0.0, 1.0 - np.abs(x - knots[i]) / h)
        
    # phi_{n_knots-1}: right edge
    result += params[n_knots - 1] * np.maximum(0.0, 1.0 - (x_max - x) / h)
    
    # Intercept
    result += params[-1]
    
    return result
# EVOLVE-BLOCK-END
