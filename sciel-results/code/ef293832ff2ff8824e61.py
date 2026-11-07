"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses scaled inputs to improve conditioning for BFGS optimization.
    The basis functions are scaled Chebyshev-like polynomials that capture
    the essential nonlinear dynamics of an oscillator while maintaining
    good numerical properties for gradient-based optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale x and v to approximately [-1, 1] for better conditioning
    # x range: [-0.5728, 0.5275], center ~ -0.02265, half-width ~ 0.55015
    # v range: [-0.1754, 0.2821], center ~ 0.05335, half-width ~ 0.22875
    X = (x + 0.02265) / 0.55015
    V = (v - 0.05335) / 0.22875
    
    # Compute scaled polynomial basis terms
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Combine basis terms with fitted parameters
    # This captures: constant, linear x, linear v, quadratic x, 
    # cross term xv, quadratic v, cubic x, x^2 v, x v^2, cubic v
    return (p0 + 
            p1 * X + 
            p2 * V + 
            p3 * X2 + 
            p4 * XV + 
            p5 * V2 + 
            p6 * X3 + 
            p7 * X2V + 
            p8 * XV2 + 
            p9 * V3)
# EVOLVE-BLOCK-END
