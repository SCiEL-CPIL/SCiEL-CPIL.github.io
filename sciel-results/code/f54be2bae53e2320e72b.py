"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: smooth elastic-plastic with temperature-dependent yield and hardening.
    stress = E(T)*strain + A(T)*(strain^2 + eps0)^m(T) * sign_smooth(strain) + C(T)
    
    Uses a quadratic-in-strain base for the hardening term to better capture
    the nonlinear stress-strain behavior of aluminium under large plastic strain.
    """
    p0, p1 = params[0], params[1]
    p2, p3 = params[2], params[3]
    p4, p5 = params[4], params[5]
    p6, p7, p8 = params[6], params[7], params[8]
    p9 = params[9]
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * temp
    A_T = p2 + p3 * temp
    m_T = np.maximum(p4 + p5 * temp, 0.1)
    C_T = p6 + p7 * temp + p8 * temp * temp
    
    # Small offset for smoothness
    eps0 = np.exp(np.clip(p9, -30, 30))
    
    # Quadratic strain term with offset: (strain^2 + eps0)^m
    strain_sq = strain * strain
    power_term = np.power(strain_sq + eps0, m_T)
    
    # Smooth sign: use strain itself as a smooth odd function multiplier
    # This naturally gives the correct sign and is smooth everywhere
    # Multiply by strain to get odd symmetry: strain * (strain^2 + eps0)^m
    hardening = A_T * strain * power_term
    
    # Combine
    elastic = E_T * strain
    result = elastic + hardening + C_T
    
    return result
# EVOLVE-BLOCK-END
