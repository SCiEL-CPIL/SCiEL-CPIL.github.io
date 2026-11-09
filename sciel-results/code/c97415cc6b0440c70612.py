"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalized basis functions, each O(1) on [-3, 3]
    x_safe = np.clip(x, -10, 10)  # safety for extreme values
    
    # Basis 0: constant
    b0 = np.ones_like(x_safe)
    
    # Basis 1: linear, normalized
    b1 = x_safe / 3.0
    
    # Basis 2: quadratic, normalized
    b2 = (x_safe ** 2) / 9.0
    
    # Basis 3: cubic, normalized
    b3 = (x_safe ** 3) / 27.0
    
    # Basis 4: Gaussian bump
    b4 = np.exp(-(x_safe ** 2))
    
    # Basis 5: tanh
    b5 = np.tanh(x_safe)
    
    # Basis 6: sine
    b6 = np.sin(x_safe)
    
    # Basis 7: Lorentzian-like
    b7 = 1.0 / (1.0 + x_safe ** 2)
    
    # Basis 8: odd Lorentzian-like
    b8 = x_safe / (1.0 + x_safe ** 2)
    
    # Basis 9: exponential decay
    b9 = np.exp(-np.abs(x_safe))
    
    # Combine with parameters
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
    
    return pred
# EVOLVE-BLOCK-END
