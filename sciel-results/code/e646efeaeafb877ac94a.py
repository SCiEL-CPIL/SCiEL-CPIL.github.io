import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a multiplicative temperature-modulated
    elastic-plastic model with smooth hardening.
    
    The model:
    - Computes temperature-dependent elastic modulus E(T) and yield stress sigma_y(T)
      using low-order polynomials in temp.
    - Applies a smooth exponential saturation / sinh-based hardening function to strain
      beyond the yield point, capturing plastic behavior.
    
    params (10 constants):
      p0, p1, p2 : coefficients for E(T) = p0 + p1*temp + p2*temp^2
      p3, p4, p5 : coefficients for sigma_y(T) = p3 + p4*temp + p5*temp^2
      p6 : hardening rate parameter
      p7 : saturation stress scale
      p8 : transition width (controls sharpness of elastic-plastic transition)
      p9 : offset / baseline stress
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * T + p2 * T * T
    
    # Temperature-dependent yield stress
    sigma_y = p3 + p4 * T + p5 * T * T
    
    # Elastic stress component
    stress_elastic = E * eps
    
    # Plastic strain beyond yield (smooth approximation)
    # Use a smooth max-like function: for eps > sigma_y/E, plastic strain activates
    eps_yield = sigma_y / (E + 1e-12)  # avoid division by zero
    # Smooth transition using tanh or similar
    # plastic_strain = max(0, eps - eps_yield) smoothed
    diff = eps - eps_yield
    # Smooth positive part using: 0.5 * (diff + |diff|) approximated by tanh
    # Use a softplus-like function: (1/p8) * log(1 + exp(p8 * diff)) - (1/p8)*log(2)
    # But to keep it simple and differentiable, use:
    # softplus(x) = log(1 + exp(x)) / scale
    # Let's use a tanh-based smooth max: 0.5*(x + x*|x|/(x*|x|+1)) is messy
    # Simpler: use np.where with a smooth blend
    
    # Use a smooth approximation of max(0, x):
    # smooth_max(0, x) = 0.5 * (x + np.sqrt(x*x + 1/p8**2)) - 0.5/p8
    # This is the softplus variant
    inv_p8 = 1.0 / (p8 + 1e-12)
    smooth_max = 0.5 * (diff + np.sqrt(diff * diff + inv_p8 * inv_p8)) - 0.5 * inv_p8
    plastic_strain = np.maximum(smooth_max, 0.0)
    
    # Hardening: exponential saturation
    # sigma_plastic = sigma_y * (1 - exp(-p6 * plastic_strain)) or similar
    # Use: sigma_plastic = p7 * (1 - exp(-p6 * plastic_strain))
    hardening = p7 * (1.0 - np.exp(-p6 * plastic_strain))
    
    # Total stress
    stress = stress_elastic + hardening + p9
    
    return stress
