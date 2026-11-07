# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid fixed-basis model using Chebyshev polynomials and trigonometric functions.
    
    This approach combines the stability of a linear least-squares problem (ideal for BFGS)
    with a basis set capable of representing smooth, polynomial, and periodic functions.
    
    Parameters:
    - p[0:6] : Coefficients for Chebyshev polynomials T_0 to T_5 (mapped to [-1, 1])
    - p[6]   : Coefficient for sin(x)
    - p[7]   : Coefficient for cos(x)
    - p[8]   : Coefficient for sin(2x)
    - p[9]   : Coefficient for cos(2x)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Normalize x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Calculate Chebyshev Polynomials T_0 to T_5
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t * t - 1.0
    b3 = 4.0 * b1 * b1 * b1 - 3.0 * b1
    b4 = 8.0 * b1 * b1 * b1 * b1 - 8.0 * b1 * b1 + 1.0
    b5 = 16.0 * b1 * b1 * b1 * b1 * b1 - 20.0 * b1 * b1 * b1 + 5.0 * b1
    
    # Calculate Trigonometric terms
    # Note: Using x (original scale) for trig functions as frequencies are natural in the original domain
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.sin(2.0 * x)
    b9 = np.cos(2.0 * x)
    
    # Linear combination using the 10 parameters
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
