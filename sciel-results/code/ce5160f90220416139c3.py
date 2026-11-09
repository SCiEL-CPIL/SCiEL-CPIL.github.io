# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use Chebyshev polynomials for better conditioning.
    # Scale x to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # Recurrence
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    T5 = 2 * t * T4 - T3
    T6 = 2 * t * T5 - T4
    T7 = 2 * t * T6 - T5
    T8 = 2 * t * T7 - T6
    T9 = 2 * t * T8 - T7
    
    # Linear combination
    y = (params[0] * T0 + 
         params[1] * T1 + 
         params[2] * T2 + 
         params[3] * T3 + 
         params[4] * T4 + 
         params[5] * T5 + 
         params[6] * T6 + 
         params[7] * T7 + 
         params[8] * T8 + 
         params[9] * T9)
         
    return y
# EVOLVE-BLOCK-END
