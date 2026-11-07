"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a flexible additive basis expansion.
    
    The model uses biologically motivated basis functions:
    - Monod kinetics for substrate saturation.
    - Gaussian functions for temperature and pH optima.
    - Inverse density for population inhibition.
    - Interaction terms to capture non-separable effects.
    
    The model is linear in `params`, ensuring the optimization landscape is convex
    for the BFGS solver.
    """
    # Calculate basis functions
    # 1. Substrate saturation (Monod)
    f_s = s / (1.0 + s)
    
    # 2. Temperature dependence (Gaussian, optimum ~37C)
    f_t = np.exp(-((temp - 37.0) ** 2) / 50.0)
    
    # 3. pH dependence (Gaussian, optimum ~7)
    f_p = np.exp(-((pH - 7.0) ** 2) / 8.0)
    
    # 4. Density inhibition
    f_b = 1.0 / (1.0 + b)
    
    # Construct the linear combination of basis functions
    # params[0]: Bias
    # params[1]: Substrate effect
    # params[2]: Temperature effect
    # params[3]: pH effect
    # params[4]: Density effect
    # params[5]: Substrate-Temp interaction
    # params[6]: Substrate-pH interaction
    # params[7]: Temp-pH interaction
    # params[8]: Full multiplicative interaction
    # params[9]: Linear temperature correction
    
    pred = (
        params[0] * 1.0 +
        params[1] * f_s +
        params[2] * f_t +
        params[3] * f_p +
        params[4] * f_b +
        params[5] * (f_s * f_t) +
        params[6] * (f_s * f_p) +
        params[7] * (f_t * f_p) +
        params[8] * (f_s * f_t * f_p * f_b) +
        params[9] * temp
    )
    
    return pred
# EVOLVE-BLOCK-END
