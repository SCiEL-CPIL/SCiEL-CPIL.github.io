"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Physics-informed Duffing-van der Pol structure with 10 parameters.
    
    a = p0*x + p1*x^3 + p2*v + p3*v^3 + p4*x*v^2 + p5*v^2*x^2 
        + p6*x^2*v + p7*x^4 + p8*v^4 + p9
    """
    # Precompute powers
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    x4 = x2 * x2
    v4 = v2 * v2
    
    # Construct the nonlinear oscillator model
    # Duffing: p0*x + p1*x^3 + p7*x^4
    # van der Pol damping: p2*v + p3*v^3
    # Cross terms: p4*x*v^2 + p5*v^2*x^2 + p6*x^2*v
    # Higher velocity: p8*v^4
    # Constant offset: p9
    term0 = params[0] * x
    term1 = params[1] * x3
    term2 = params[2] * v
    term3 = params[3] * v3
    term4 = params[4] * x * v2
    term5 = params[5] * v2 * x2
    term6 = params[6] * x2 * v
    term7 = params[7] * x4
    term8 = params[8] * v4
    term9 = params[9] * np.ones_like(x)
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
