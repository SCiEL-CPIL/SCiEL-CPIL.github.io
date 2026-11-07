"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a flexible basis expansion.
    
    Basis functions:
    0: constant
    1: b (density)
    2: s (substrate)
    3: s / (s + Ks)  -- Monod saturation
    4: temp
    5: (temp - T_opt)^2  -- quadratic temp penalty
    6: pH
    7: (pH - pH_opt)^2  -- quadratic pH penalty
    8: b * s / (s + Ks)  -- density * Monod
    9: temp * pH  -- interaction
    
    We use 10 params. The Ks, T_opt, pH_opt are fixed to reasonable biological values
    to keep the expression closed-form with only linear params.
    """
    # Fixed biological constants for saturation/penalty centers
    Ks = 10.0
    T_opt = 37.0
    pH_opt = 7.0
    
    # Monod saturation term
    monod = s / (s + Ks)
    
    # Quadratic penalties
    temp_sq = (temp - T_opt) ** 2
    ph_sq = (pH - pH_opt) ** 2
    
    # Build basis
    term0 = np.ones_like(b)
    term1 = b
    term2 = s
    term3 = monod
    term4 = temp
    term5 = temp_sq
    term6 = pH
    term7 = ph_sq
    term8 = b * monod
    term9 = temp * pH
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
