"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a mixed basis of Chebyshev polynomials,
    trigonometric functions, and nonlinear activation functions.
    
    Basis functions (10 total):
    0: T_0(t) = 1
    1: T_1(t) = t
    2: T_2(t) = 2t^2 - 1
    3: T_3(t) = 4t^3 - 3t
    4: sin(x)
    5: cos(x)
    6: sin(2x)
    7: cos(2x)
    8: exp(-x^2)
    9: tanh(x)
    
    where t = x / 3.0 maps x from [-3, 3] to [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.flatten()
    
    if x_flat.size == 0:
        return np.array([]).reshape(original_shape)
        
    # Map to [-1, 1] for Chebyshev stability
    t = x_flat / 3.0
    
    # Chebyshev polynomials T_0 to T_3
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    
    # Trigonometric functions
    S1 = np.sin(x_flat)
    C1 = np.cos(x_flat)
    S2 = np.sin(2.0 * x_flat)
    C2 = np.cos(2.0 * x_flat)
    
    # Nonlinear functions
    E1 = np.exp(-x_flat * x_flat)
    H1 = np.tanh(x_flat)
    
    # Linear combination
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * S1 +
         params[5] * C1 +
         params[6] * S2 +
         params[7] * C2 +
         params[8] * E1 +
         params[9] * H1)
         
    return y.reshape(original_shape)
# EVOLVE-BLOCK-END
