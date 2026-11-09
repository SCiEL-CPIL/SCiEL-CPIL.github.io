"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a composite model:
    - Elastic region: linear in strain, modulated by temperature
    - Plastic region: power-law hardening with temperature softening
    - Smooth transition between elastic and plastic using a sigmoid-like function
    
    params (10 constants):
      p0: elastic modulus base
      p1: elastic modulus temp coefficient
      p2: yield strain offset
      p3: hardening exponent
      p4: hardening coefficient
      p5: temp softening coefficient for yield
      p6: temp softening coefficient for hardening
      p7: transition sharpness
      p8: residual stress offset
      p9: additional temp effect
    """
    # Extract parameters
    E0 = params[0]       # Base elastic modulus
    E_temp = params[1]   # Temperature effect on modulus
    eps_y = params[2]    # Yield strain
    n_hard = params[3]   # Hardening exponent
    K_hard = params[4]   # Hardening coefficient
    T_soft_y = params[5] # Temp softening on yield
    T_soft_h = params[6] # Temp softening on hardening
    sharp = params[7]    # Transition sharpness
    sigma_res = params[8]# Residual stress
    T_extra = params[9]  # Additional temperature effect
    
    # Ensure positive values where needed by using absolute value or softplus-like
    # Use max with a small epsilon to avoid division by zero or negative bases
    eps_y_safe = np.maximum(np.abs(eps_y), 1e-6)
    n_hard_safe = np.maximum(np.abs(n_hard), 0.01)
    K_hard_safe = np.abs(K_hard)
    sharp_safe = np.maximum(np.abs(sharp), 1.0)
    
    # Temperature factor: reduces yield and hardening with increasing temp
    # temp is in [0.06667, 1]
    temp_factor_y = 1.0 - T_soft_y * (temp - 0.06667) / (1.0 - 0.06667)
    temp_factor_h = 1.0 - T_soft_h * (temp - 0.06667) / (1.0 - 0.06667)
    
    # Ensure temp factors are positive
    temp_factor_y = np.maximum(temp_factor_y, 0.1)
    temp_factor_h = np.maximum(temp_factor_h, 0.1)
    
    # Effective yield strain (reduced by temperature)
    eps_y_eff = eps_y_safe * temp_factor_y
    
    # Effective modulus
    E_eff = E0 + E_temp * (temp - 0.5)
    E_eff = np.maximum(np.abs(E_eff), 1e-6)
    
    # Strain relative to yield
    strain_rel = (strain - eps_y_eff) / eps_y_safe
    
    # Smooth transition function: sigmoid-like
    # For strain < eps_y_eff: elastic
    # For strain > eps_y_eff: plastic with hardening
    # Use a smooth blend based on tanh
    
    # Elastic stress
    sigma_elastic = E_eff * strain
    
    # Plastic stress: power law hardening
    # sigma_plastic = K_hard_safe * temp_factor_h * (strain_rel / eps_y_safe)^(n_hard_safe)
    # But we need to handle negative strain_rel
    strain_pos = np.maximum(strain_rel, 0.0)
    sigma_plastic = K_hard_safe * temp_factor_h * (strain_pos + 1e-6) ** n_hard_safe
    
    # Blend between elastic and plastic using a smooth function
    # t = 0.5 * (1 + tanh(sharp_safe * strain_rel))
    t = 0.5 * (1.0 + np.tanh(sharp_safe * strain_rel))
    
    # Combined stress
    sigma = (1.0 - t) * sigma_elastic + t * sigma_plastic
    
    # Add residual stress and additional temperature effect
    sigma = sigma + sigma_res + T_extra * (temp - 0.5)
    
    return sigma
# EVOLVE-BLOCK-END
