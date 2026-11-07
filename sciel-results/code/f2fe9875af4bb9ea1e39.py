"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev polynomials T_0 to T_5
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    
    # Linear combination of Chebyshev terms
    y_cheb = (params[0] * T0 + 
              params[1] * T1 + 
              params[2] * T2 + 
              params[3] * T3 + 
              params[4] * T4 + 
              params[5] * T5)
    
    # Two Sinusoidal terms to capture more complex periodic behavior
    # Term 1: A1 * sin(w1 * x)
    A1 = params[6]
    w1 = np.tanh(params[7]) * 5.0
    y_sin1 = A1 * np.sin(w1 * x)
    
    # Term 2: A2 * sin(w2 * x)
    A2 = params[8]
    w2 = np.tanh(params[9]) * 5.0
    y_sin2 = A2 * np.sin(w2 * x)
    
    return y_cheb + y_sin1 + y_sin2
# EVOLVE-BLOCK-END
