"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a combination of polynomial and smooth nonlinear basis functions
    to capture the elastic-plastic transition, work hardening, and
    temperature dependence in the stress-strain response.
    
    The model combines:
    - Linear elastic term
    - Power-law plastic hardening (smooth via softplus-like)
    - Temperature-dependent modulus reduction
    - Yield point smoothing via tanh
    - Higher-order corrections
    
    params (10 constants):
      p0: Intercept
      p1: Elastic modulus coefficient
      p2: Yield strain scale (controls transition location)
      p3: Yield stress amplitude
      p4: Hardening exponent (power-law shape)
      p5: Temperature effect on modulus
      p6: Temperature effect on yield stress
      p7: Transition sharpness
      p8: Work hardening coefficient
      p9: Higher-order strain correction
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent elastic modulus
    E_eff = p1 * (1.0 + p5 * (T - 0.5))  # Center T around 0.5 for stability
    
    # Effective strain beyond elastic limit
    # Use softplus-like function for smooth transition
    # For small x: softplus(x) ~ x; for large x: softplus(x) ~ x - log(2)
    # We want plastic strain to start after elastic limit
    eps_plastic = eps - p2 * (1.0 + p6 * (T - 0.5))
    
    # Smooth positive part: max(0, x) approximated smoothly
    # Using: 0.5 * (x + sqrt(x^2 + delta^2)) where delta controls smoothness
    delta = 1.0 / (p7 + 1e-6)  # Avoid division by zero
    smooth_pos = 0.5 * (eps_plastic + np.sqrt(eps_plastic * eps_plastic + delta * delta))
    
    # Power-law hardening: sigma_plastic = p3 * (smooth_pos)^p4
    # Clip to avoid negative base with fractional exponent
    base = np.maximum(smooth_pos, 1e-10)
    # Use log-exp for numerical stability with fractional powers
    log_base = np.log(base)
    plastic_stress = p3 * np.exp(p4 * log_base)
    
    # Elastic stress
    elastic_stress = E_eff * eps
    
    # Work hardening correction (linear in plastic strain)
    hardening = p8 * smooth_pos
    
    # Higher-order strain correction
    correction = p9 * eps * eps * (1.0 + 0.5 * T)
    
    # Combine all terms
    sigma = p0 + elastic_stress + plastic_stress + hardening + correction
    
    return sigma
# EVOLVE-BLOCK-END
