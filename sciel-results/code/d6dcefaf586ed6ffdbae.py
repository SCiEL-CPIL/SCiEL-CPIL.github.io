"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # 10 diverse basis functions
    b0 = np.ones_like(x)           # constant
    b1 = x                         # linear
    b2 = x**2                      # quadratic
    b3 = x**3                      # cubic
    b4 = np.sin(x)                 # sine
    b5 = np.cos(x)                 # cosine
    b6 = np.tanh(x)                # hyperbolic tangent
    b7 = np.exp(-x**2)            # Gaussian
    b8 = np.abs(x)                 # absolute value
    b9 = x * np.sin(x)            # x*sin(x)
    
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
# EVOLVE-BLOCK-END
