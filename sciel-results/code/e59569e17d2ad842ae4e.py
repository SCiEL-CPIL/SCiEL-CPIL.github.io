"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Self-sustained oscillator with limit cycle behavior.
    
    Model: a = p1*x + p2*x^3 + p3*v + p4*v^3 + p5*x*v + p6
    
    This combines:
    - Duffing-type position nonlinearity: p1*x + p2*x^3
    - Van der Pol-style velocity terms: p3*v + p4*v^3
    - Cross term: p5*x*v
    - Constant offset: p6
    
    Unused parameters (p7-p9) are set to zero to avoid affecting the fit.
    """
    # Extract parameters
    p1 = params[0]
    p2 = params[1]
    p3 = params[2]
    p4 = params[3]
    p5 = params[4]
    p6 = params[5]
    
    # Compute nonlinear oscillator acceleration
    x2 = x * x
    v2 = v * v
    
    # Duffing potential terms
    duffing = p1 * x + p2 * x * x2
    
    # Van der Pol damping terms (negative linear damping + cubic saturation)
    damping = p3 * v + p4 * v * v2
    
    # Cross coupling term
    cross = p5 * x * v
    
    # Constant bias
    bias = p6
    
    # Total acceleration
    a = duffing + damping + cross + bias
    
    return a
# EVOLVE-BLOCK-END
