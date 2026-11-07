"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for better numerical conditioning
    u = x / 3.0
    
    # Use a low-degree polynomial basis (up to cubic)
    # This captures mild non-linearity while remaining robust for BFGS
    # from the initial point of all ones.
    
    # u^0 = 1
    # u^1 = u
    # u^2 = u^2
    # u^3 = u^3
    
    y = (params[0] * 1.0 + 
         params[1] * u + 
         params[2] * u**2 + 
         params[3] * u**3)
         
    return y
# EVOLVE-BLOCK-END
