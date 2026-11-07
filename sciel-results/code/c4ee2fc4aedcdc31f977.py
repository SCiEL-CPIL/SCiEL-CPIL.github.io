import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Structural change: Rational Function (Padé-like)
    # This allows for saturation and inhibition effects that pure polynomials may struggle with.
    # Numerator: Linear combination of inputs
    # Denominator: 1 + Linear combination + Quadratic term for temperature (inhibition at high temp?)
    
    num = (p[0] 
           + p[1] * b_norm 
           + p[2] * s_norm 
           + p[3] * t_norm 
           + p[4] * ph_norm)
           
    den = (1.0 
           + p[5] * b_norm 
           + p[6] * s_norm 
           + p[7] * t_norm 
           + p[8] * ph_norm 
           + p[9] * t_norm**2)
           
    # Safety: ensure denominator is not close to zero to avoid instability
    # Since inputs are [0,1] and params start at 1, den starts at 6.
    # We clip den to avoid division by zero if optimizer wanders.
    den_safe = np.where(np.abs(den) < 1e-6, 1e-6, den)
    
    y = num / den_safe
    
    return y
