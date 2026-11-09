"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a well-conditioned set of basis functions covering common symbolic regression targets
    # Polynomials up to degree 4, trig functions, Gaussian, and tanh
    # Scale x to [-1, 1] range for better conditioning
    
    x_scaled = x / 3.0  # Map [-3,3] to [-1,1]
    
    b0 = np.ones_like(x_scaled)
    b1 = x_scaled
    b2 = x_scaled ** 2
    b3 = x_scaled ** 3
    b4 = x_scaled ** 4
    b5 = np.sin(np.pi * x_scaled)
    b6 = np.cos(np.pi * x_scaled)
    b7 = np.sin(2 * np.pi * x_scaled)
    b8 = np.exp(-x_scaled ** 2)
    b9 = np.tanh(x_scaled)
    
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
