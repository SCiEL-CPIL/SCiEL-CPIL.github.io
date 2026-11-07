"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining polynomial basis with Monod gating.
    
    The model uses:
    - Monod-like substrate saturation (u_s)
    - Quadratic polynomial in temperature, gated by substrate
    - Quadratic polynomial in pH
    - Linear density inhibition
    - Interaction terms: substrate*pH, temp*pH, substrate*temp
    
    This is linear in params, giving a convex optimization landscape for BFGS.
    """
    # Monod-like substrate saturation
    u_s = s / (1.0 + s)
    
    # Temperature polynomial (quadratic), gated by substrate
    t_poly = params[1] + params[2] * temp + params[3] * temp**2
    y_temp = u_s * t_poly
    
    # pH polynomial (quadratic)
    y_pH = params[4] + params[5] * pH + params[6] * pH**2
    
    # Density effect (linear)
    y_b = params[7] * b
    
    # Interactions
    y_s_pH = params[8] * u_s * pH
    y_T_pH = params[9] * temp * pH
    
    # Combine all terms with bias
    pred = params[0] + y_temp + y_pH + y_b + y_s_pH + y_T_pH
    
    return pred
# EVOLVE-BLOCK-END
