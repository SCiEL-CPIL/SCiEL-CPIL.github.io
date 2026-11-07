import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a hybrid Voce-Ramberg-Osgood formulation.
    
    This model combines the asymptotic saturation of the Voce model with the smooth
    yield-point transition of a Ramberg-Osgood power-law term. It uses centered strain
    for the linear and polynomial components to improve conditioning, while using
    absolute strain for the nonlinear hardening term to handle compression correctly.
    
    sigma = E(T)*u_c + sigma_y(T) + Q(T)*(1 - exp(-u/H(T))) + K(T)*|u|^n(T) + c(T)*u_c^2
    
    Where:
      u_c = strain - 0.5  (Centered strain for linear/polynomial terms)
      E(T) = p0 + p1*T          (Elastic modulus)
      sigma_y(T) = p2 + p3*T    (Yield stress)
      Q(T) = p4 + p5*T          (Voce saturation stress range)
      H(T) = softplus(p6) + softplus(p7)*T  (Voce hardening modulus, constrained positive)
      K(T) = p8 + p9*T          (Ramberg-Osgood hardening coefficient)
      
    Note: The exponent n for the power-law term is fixed at 2.0 (quadratic) to conserve
    parameter budget and maintain smoothness, effectively acting as a higher-order
    correction that is distinct from the centered quadratic term c(T)*u_c^2 due to the
    non-centering of the |u|^2 term. This provides a specific non-symmetric correction
    in the positive strain region.
    
    Parameters (10):
    p0, p1: Elastic modulus coefficients
    p2, p3: Yield stress coefficients
    p4, p5: Voce saturation stress coefficients
    p6, p7: Voce hardening modulus coefficients (via softplus to ensure positivity)
    p8, p9: Power-law correction coefficients
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Helper for stable softplus to avoid overflow/underflow and ensure differentiability
    def stable_softplus(x):
        # For x > 20, softplus(x) ~ x
        # For x <= 20, softplus(x) = log(1 + exp(x))
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 50))))
    
    # Center strain to improve conditioning of polynomial-like terms
    # Range of u is approx [-0.002, 0.964], so center at 0.5
    u_c = u - 0.5
    
    # Temperature-dependent coefficients
    # E(T): Elastic modulus
    E = p0 + p1 * T
    
    # sigma_y(T): Yield stress
    sigma_y = p2 + p3 * T
    
    # Q(T): Voce saturation stress range
    Q = p4 + p5 * T
    
    # H(T): Voce hardening modulus (constrained positive via softplus)
    H_const = stable_softplus(p6)
    H_temp_coeff = stable_softplus(p7)
    H = H_const + H_temp_coeff * T
    
    # Ensure H is not too small to avoid numerical issues
    H_safe = np.maximum(H, 1e-6)
    
    # Voce hardening term: Q * (1 - exp(-u/H))
    # Use original u because exponential decay is physically tied to absolute strain
    arg = -u / H_safe
    arg_clipped = np.clip(arg, -50, 50)
    voce_term = Q * (1.0 - np.exp(arg_clipped))
    
    # Elastic contribution (using centered strain for better orthogonality)
    elastic = E * u_c
    
    # Power-law correction term (Ramberg-Osgood style)
    # Using absolute value to handle negative strains (compression)
    # Exponent is fixed at 2.0 for stability and smoothness
    u_abs = np.abs(u)
    # Clip to avoid 0^0 or numerical issues, though 2.0 is safe
    u_clipped = np.clip(u_abs, 1e-10, 10.0)
    # K(T) * |u|^2
    power_term = (p8 + p9 * T) * (u_clipped * u_clipped)
    
    # Combine all terms
    # Note: We do not add a separate centered quadratic term (p8+p9)*u_c^2 
    # because we are using p8, p9 for the power term. 
    # The combination of elastic (linear in u_c) and power (quadratic in |u|) 
    # provides sufficient flexibility. The Voce term handles the saturation.
    
    sigma = elastic + sigma_y + voce_term + power_term
    
    return sigma
