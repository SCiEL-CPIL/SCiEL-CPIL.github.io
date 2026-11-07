"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev polynomials T_0 to T_5
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t^2 - 1
    # T_3 = 4t^3 - 3t
    # T_4 = 8t^4 - 8t^2 + 1
    # T_5 = 16t^5 - 20t^3 + 5t
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    
    # Linear combination of Chebyshev terms
    # params[0] to params[5]
    y_cheb = (params[0] * T0 + 
              params[1] * T1 + 
              params[2] * T2 + 
              params[3] * T3 + 
              params[4] * T4 + 
              params[5] * T5)
    
    # Sinusoidal term: A * sin(w * x)
    # params[6] is amplitude A
    # params[7] is raw frequency, mapped to [-5, 5] via tanh
    A = params[6]
    w = np.tanh(params[7]) * 5.0
    y_sin = A * np.sin(w * x)
    
    # Exponential term: B * exp(C * x)
    # params[8] is amplitude B
    # params[9] is raw rate, mapped to [-2, 2] via tanh to prevent overflow
    B = params[8]
    C = np.tanh(params[9]) * 2.0
    y_exp = B * np.exp(C * x)
    
    return y_cheb + y_sin + y_exp
# EVOLVE-BLOCK-END
