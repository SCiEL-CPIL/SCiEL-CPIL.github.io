"""Symbolic regression: stress-strain of an aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Smoothed piecewise linear model for stress-strain behavior.
    
    The model assumes a transition from elastic to plastic behavior.
    Elastic stress: sigma_e = E(t) * e
    Plastic stress: sigma_p = sigma_y(t) + H(t) * (e - e_y(t))
    
    Where:
    E(t) = p[0] + p[1]*t + p[2]*t^2
    e_y(t) = p[8] + p[9]*t
    sigma_y(t) = E(t) * e_y(t)
    H(t) = p[6] + p[7]*t
    
    The total stress is the smooth maximum of sigma_e and sigma_p.
    smoothmax(a, b) = (a + b + sqrt((a - b)^2 + delta^2)) / 2
    
    Parameters:
    p[0], p[1], p[2]: Elastic modulus coefficients (constant, linear, quadratic in temp)
    p[3], p[4], p[5]: (Unused in this specific formulation, reserved for future expansion or 
                       could be used to offset sigma_y directly if E*ey is not sufficient. 
                       Let's use them to make sigma_y independent: 
                       sigma_y(t) = p[3] + p[4]*t + p[5]*t^2)
    p[6], p[7]: Hardening modulus coefficients (constant, linear in temp)
    p[8], p[9]: Yield strain coefficients (constant, linear in temp)
    
    Revised Parameter Mapping:
    p[0], p[1], p[2]: E(t) = p[0] + p[1]*t + p[2]*t^2
    p[3], p[4], p[5]: sigma_y(t) = p[3] + p[4]*t + p[5]*t^2
    p[6], p[7]: H(t) = p[6] + p[7]*t
    p[8], p[9]: e_y(t) = p[8] + p[9]*t
    
    This uses all 10 parameters effectively.
    """
    e = strain
    t = temp
    p = params

    # Temperature-dependent Elastic Modulus E(t)
    E_t = p[0] + p[1] * t + p[2] * t**2
    
    # Temperature-dependent Yield Stress sigma_y(t)
    sigma_y_t = p[3] + p[4] * t + p[5] * t**2
    
    # Temperature-dependent Hardening Modulus H(t)
    H_t = p[6] + p[7] * t
    
    # Temperature-dependent Yield Strain e_y(t)
    e_y_t = p[8] + p[9] * t

    # Elastic stress
    sigma_e = E_t * e
    
    # Plastic stress
    # Note: Plastic stress should only be active for e > e_y, 
    # but smoothmax handles the blending. 
    # If e < e_y, (e - e_y) is negative, so sigma_p might be lower than sigma_e.
    sigma_p = sigma_y_t + H_t * (e - e_y_t)

    # Smooth maximum to ensure differentiability
    # delta controls the sharpness of the transition. 
    # A small value like 1e-4 approximates max while staying differentiable.
    delta = 1e-4
    diff = sigma_e - sigma_p
    smooth_max = 0.5 * (sigma_e + sigma_p + np.sqrt(diff**2 + delta**2))

    return smooth_max
# EVOLVE-BLOCK-END
