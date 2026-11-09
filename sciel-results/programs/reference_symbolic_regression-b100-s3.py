"""Rewritten program for symbolic regression with flexible non-linear basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model combining polynomial, sinusoidal, and exponential terms.
    
    params (length 10):
        p0: coefficient for constant
        p1: coefficient for x
        p2: coefficient for x^2
        p3: coefficient for x^3
        p4: coefficient for x^4
        p5: amplitude for sin(freq*x)
        p6: frequency for sin(freq*x)
        p7: amplitude for cos(freq*x)
        p8: coefficient for exp(x)
        p9: coefficient for tanh(x)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Polynomial terms (up to degree 4)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Sinusoidal terms with learnable frequency
    freq = params[6]
    # Clamp frequency to avoid numerical issues
    freq_clamped = np.clip(freq, -10.0, 10.0)
    sin_term = np.sin(freq_clamped * x)
    cos_term = np.cos(freq_clamped * x)
    
    # Exponential term with clamping to avoid overflow
    exp_arg = np.clip(x, -10.0, 10.0)
    exp_term = np.exp(exp_arg)
    
    # Tanh term
    tanh_term = np.tanh(x)
    
    # Combine all terms
    result = (
        params[0]
        + params[1] * x
        + params[2] * x2
        + params[3] * x3
        + params[4] * x4
        + params[5] * sin_term
        + params[7] * cos_term
        + params[8] * exp_term
        + params[9] * tanh_term
    )
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
