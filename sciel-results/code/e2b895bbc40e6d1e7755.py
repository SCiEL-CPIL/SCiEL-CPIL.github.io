"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Self-sustained oscillator with limit cycle behavior.
    
    Model: a = p1*x + p2*x^3 + p3*v + p4*v^2 + p5*v^3 + p6*x*v + p7*x^2 + p8*sin(w*t) + p9*x*sin(w*t) + p10
    
    This combines:
    - Duffing-type position nonlinearity: p1*x + p2*x^3
    - Van der Pol-style velocity terms: p3*v + p4*v^2 + p5*v^3
    - Cross term: p6*x*v
    - Quadratic position: p7*x^2
    - Sinusoidal driving: p8*sin(w*t)
    - Parametric excitation: p9*x*sin(w*t)
    - Constant offset: p10
    """
    # Extract parameters
    p1 = params[0]
    p2 = params[1]
    p3 = params[2]
    p4 = params[3]
    p5 = params[4]
    p6 = params[5]
    p7 = params[6]
    p8 = params[7]
    p9 = params[8]
    p10 = params[9]
    
    # Compute nonlinear oscillator acceleration
    x2 = x * x
    v2 = v * v
    
    # Duffing potential terms
    duffing = p1 * x + p2 * x * x2
    
    # Van der Pol damping terms (linear + quadratic + cubic)
    damping = p3 * v + p4 * v2 + p5 * v * v2
    
    # Cross coupling term
    cross = p6 * x * v
    
    # Quadratic position nonlinearity
    quad_pos = p7 * x2
    
    # Sinusoidal driving term (fixed frequency ~1 rad/s based on typical oscillator)
    sin_t = np.sin(t)
    drive = p8 * sin_t
    
    # Parametric excitation
    param_exc = p9 * x * sin_t
    
    # Constant bias
    bias = p10
    
    # Total acceleration
    a = duffing + damping + cross + quad_pos + drive + param_exc + bias
    
    return a
# EVOLVE-BLOCK-END
