"""
Direct approach: build a flexible model from the task structure.
The task asks for symbolic regression with 1 input on [-3, 3].
We design a basis that covers common smooth functions BFGS can fit with 10 params.
Uses Chebyshev polynomials for better conditioning.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model with 10 parameters:
    - Chebyshev polynomial terms (degree 0-4) scaled to [-1,1]: 5 params
    - Trigonometric terms (sin/cos of x, 2x): 4 params
    - One exponential/rational term: 1 param
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    x_scaled = x / 3.0
    
    # Chebyshev polynomials of the first kind T_n(x)
    # T_0 = 1
    # T_1 = x
    # T_2 = 2x^2 - 1
    # T_3 = 4x^3 - 3x
    # T_4 = 8x^4 - 8x^2 + 1
    
    T0 = np.ones_like(x_scaled)
    T1 = x_scaled
    T2 = 2 * x_scaled**2 - 1
    T3 = 4 * x_scaled**3 - 3 * x_scaled
    T4 = 8 * x_scaled**4 - 8 * x_scaled**2 + 1
    
    # Polynomial part using Chebyshev basis
    result = p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3 + p[4] * T4
    
    # Trigonometric part: p5*sin(x) + p6*cos(x) + p7*sin(2x) + p8*cos(2x)
    result += p[5] * np.sin(x) + p[6] * np.cos(x)
    result += p[7] * np.sin(2 * x) + p[8] * np.cos(2 * x)
    
    # Exponential decay: p9 * exp(-x^2)
    result += p[9] * np.exp(-x**2)
    
    return result
# EVOLVE-BLOCK-END
