# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates the model using a rich basis of Chebyshev polynomials,
    trigonometric, hyperbolic, and Gaussian functions.
    params[0..9] are coefficients for:
    1, T1(x/3), T2(x/3), T3(x/3), T4(x/3), T5(x/3), T6(x/3), T7(x/3), sin(x), cos(x)
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute scaled variable for Chebyshev polynomials on [-1, 1]
    t = x / 3.0
    
    # Chebyshev polynomials of the first kind using recurrence
    # T0(t) = 1, T1(t) = t
    # Tn(t) = 2*t*Tn-1(t) - Tn-2(t)
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    T7 = 2.0 * t * T6 - T5
    
    # Other basis functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    # Combine
    y = (p0 * T0 +
         p1 * T1 +
         p2 * T2 +
         p3 * T3 +
         p4 * T4 +
         p5 * T5 +
         p6 * T6 +
         p7 * T7 +
         p8 * sin_x +
         p9 * cos_x)
         
    return y
# EVOLVE-BLOCK-END
