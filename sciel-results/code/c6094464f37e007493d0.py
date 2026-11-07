"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible polynomial surface for E. coli growth rate.
    Uses up to 2nd order terms in normalized inputs to capture complex interactions
    without assuming specific saturation or peaked shapes.
    """
    # Normalize inputs to roughly [0, 1] range based on training data ranges
    # b: [1, 4.999], s: [1, 39.98], temp: [1, 12], pH: [1, 4.999] (approx)
    # Using conservative normalization to avoid large values
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Create feature matrix with up to 2nd order polynomial terms
    # We have 4 inputs, so 1st order: 4 terms, 2nd order: 4*5/2 = 10 terms, plus intercept = 1
    # Total: 15 terms. But we only have 10 params.
    # Let's select key terms: intercept, 4 linear, 5 quadratic (diagonal + key cross terms)
    
    # Build features: [1, b, s, t, p, b^2, s^2, t^2, p^2, b*s]
    # This gives us 10 features matching 10 params
    
    ones = np.ones_like(b)
    f1 = ones
    f2 = b_norm
    f3 = s_norm
    f4 = temp_norm
    f5 = ph_norm
    f6 = b_norm ** 2
    f7 = s_norm ** 2
    f8 = temp_norm ** 2
    f9 = ph_norm ** 2
    f10 = b_norm * s_norm
    
    # Linear combination of features with params
    mu = (params[0] * f1 + 
          params[1] * f2 + 
          params[2] * f3 + 
          params[3] * f4 + 
          params[4] * f5 + 
          params[5] * f6 + 
          params[6] * f7 + 
          params[7] * f8 + 
          params[8] * f9 + 
          params[9] * f10)
    
    return mu
# EVOLVE-BLOCK-END
