"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a combination of polynomial and trigonometric basis functions
    # to capture a wider variety of potential target functions.
    # x is in [-2, 2], so trig functions with frequencies 1 and 2 are reasonable.
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    x_sin_x = x * sin_x
    
    return (params[0] * 1.0 + 
            params[1] * x + 
            params[2] * x2 + 
            params[3] * x3 + 
            params[4] * x4 + 
            params[5] * sin_x + 
            params[6] * cos_x + 
            params[7] * sin_2x + 
            params[8] * cos_2x + 
            params[9] * x_sin_x)
# EVOLVE-BLOCK-END
