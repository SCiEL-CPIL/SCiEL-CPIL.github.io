import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses exponential saturation for elastic-plastic transition
    with temperature-dependent softening and hardening terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays for broadcasting
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Safe parameter transformations to avoid numerical issues
    # Clamp exponent arguments to prevent overflow/underflow
    exp_arg = np.clip(p2 * strain * (1.0 + p3 * temp), -50.0, 50.0)
    
    # Exponential saturation term: captures yield point and strain hardening
    # At small strain: ~ p0 * strain * (1+p1*temp) * p2 * strain = linear
    # At large strain: saturates to p0 * (1+p1*temp)
    exp_term = p0 * (1.0 + p1 * temp) * (1.0 - np.exp(-exp_arg))
    
    # Polynomial terms for additional flexibility
    # Linear elastic, temperature effects, and interaction terms
    poly_term = (
        p4 * strain + 
        p5 * temp + 
        p6 * strain**2 + 
        p7 * temp * strain + 
        p8 + 
        p9 * temp**2
    )
    
    return exp_term + poly_term
