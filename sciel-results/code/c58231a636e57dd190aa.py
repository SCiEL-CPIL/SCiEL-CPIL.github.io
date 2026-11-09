import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis for nonlinear oscillator acceleration.
    
    Uses Chebyshev polynomials of degree up to 3 in both x and v,
    providing better conditioning than monomial basis for BFGS optimization.
    
    Parameters (10 total):
    p[0..3]: Coefficients for Chebyshev terms in x: T0, T1, T2, T3
    p[4..7]: Coefficients for Chebyshev terms in v: T0, T1, T2, T3
    p[8]: Coefficient for interaction term T2(x)*T2(v)
    p[9]: Coefficient for interaction term T1(x)*T1(v)
    """
    # Normalization constants based on training ranges
    # x: [-0.5728, 0.5275] -> map to [-1, 1]
    # v: [-0.1754, 0.2821] -> map to [-1, 1]
    
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Map to [-1, 1] for Chebyshev polynomials
    x_n = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    v_n = 2.0 * (v - v_min) / (v_max - v_min) - 1.0
    
    p = params
    
    # Chebyshev polynomials: T0=1, T1=t, T2=2t^2-1, T3=4t^3-3t
    T0_x = np.ones_like(x_n)
    T1_x = x_n
    T2_x = 2.0 * x_n * x_n - 1.0
    T3_x = 4.0 * x_n * x_n * x_n - 3.0 * x_n
    
    T0_v = np.ones_like(v_n)
    T1_v = v_n
    T2_v = 2.0 * v_n * v_n - 1.0
    T3_v = 4.0 * v_n * v_n * v_n - 3.0 * v_n
    
    # Linear combination of Chebyshev basis functions
    result = (p[0] * T0_x + 
              p[1] * T1_x + 
              p[2] * T2_x + 
              p[3] * T3_x +
              p[4] * T0_v + 
              p[5] * T1_v + 
              p[6] * T2_v + 
              p[7] * T3_v +
              p[8] * T2_x * T2_v +
              p[9] * T1_x * T1_v)
    
    return result
