"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a smooth piecewise model with temperature-dependent coefficients.
    
    Model:
    sigma = Elastic_Term + Plastic_Term + Correction_Term
    
    Elastic_Term = E(T) * eps * exp(-eps^2 / (2 * sigma_eps(T)^2))
    Plastic_Term = Y(T) * tanh(R(T) * (eps - eps_y(T))) * H(eps - eps_y(T))
    Correction_Term = p8 * eps^2 + p9 * eps^3
    
    Parameters (10 constants):
    p0, p1, p2: Elastic modulus coefficients (quadratic in T)
    p3, p4: Strain spread coefficients (linear in T)
    p5, p6: Yield stress coefficients (linear in T)
    p7, p8: Hardening rate coefficients (linear in T)
    p9: Cubic correction coefficient
    p10: Quadratic correction coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * T + p2 * T * T
    sigma_eps_T = np.exp(p3 + p4 * T)  # Positive definite spread parameter
    Y_T = p5 + p6 * T
    R_T = p7 + p8 * T
    
    # Smooth elastic-to-plastic transition
    # Use a smooth approximation of Heaviside step function
    eps_y = 0.001  # Yield strain
    H_smooth = 0.5 * (1 + np.tanh((eps - eps_y) / 0.0005))
    
    # Elastic term: Gaussian decay to model the transition from linear elastic to plastic
    elastic_term = E_T * eps * np.exp(-eps**2 / (2 * sigma_eps_T**2))
    
    # Plastic term: tanh-based hardening with smooth activation
    plastic_term = Y_T * np.tanh(R_T * (eps - eps_y)) * H_smooth
    
    # Correction terms
    quad_term = p8 * eps**2
    cubic_term = p9 * eps**3
    
    sigma = elastic_term + plastic_term + quad_term + cubic_term
    
    return sigma
# EVOLVE-BLOCK-END
