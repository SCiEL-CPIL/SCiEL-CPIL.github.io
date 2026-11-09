# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model: quadratic response surface + Monod substrate saturation + Gaussian temperature response.
    
    This hybrid model captures:
    - Quadratic curvature (saturation, peaks, interactions) via centered quadratic terms
    - Substrate saturation kinetics via a Monod-like term (smooth, bounded)
    - Temperature optimum via a Gaussian response (biologically motivated)
    
    The centered inputs and smooth nonlinear features create a well-conditioned loss landscape
    for BFGS optimization starting from params=ones.
    """
    p = params
    
    # Center inputs to improve conditioning and make params start near reasonable values
    b_c = b - 3.0
    s_c = s - 10.0
    t_c = temp - 30.0
    p_c = pH - 7.0
    
    # Quadratic terms (captures curvature and interactions)
    b2 = b_c * b_c
    s2 = s_c * s_c
    t2 = t_c * t_c
    p2 = p_c * p_c
    
    # Interaction terms (cross-effects between variables)
    bs = b_c * s_c
    bt = b_c * t_c
    sp = s_c * p_c
    
    # Monod-like substrate saturation: s / (K + s), where K = exp(params[7]) to keep positive
    # Use a smooth version: s_c centered, but Monod needs positive s
    # s is in [1, 4.999] range, so use s directly with a small offset for safety
    K_s = np.exp(p[7])  # Half-saturation constant, kept positive via exp
    monod_s = s / (K_s + s)
    
    # Gaussian temperature response: exp(-(temp - T_opt)^2 / (2 * sigma^2))
    # T_opt and sigma are derived from params to ensure positivity
    T_opt = 25.0 + 10.0 * np.tanh(p[8])  # Optimum temperature in [15, 35]
    sigma_t = np.exp(p[9]) + 0.1  # Temperature width, kept positive
    gauss_t = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_t * sigma_t))
    
    # Combine all features with fitted coefficients
    # params[0..6] for linear/quadratic/interaction terms
    # params[7] for Monod K_s (via exp)
    # params[8] for temperature optimum (via tanh)
    # params[9] for temperature width (via exp)
    
    pred = (
        p[0] * b_c +
        p[1] * s_c +
        p[2] * t_c +
        p[3] * p_c +
        p[4] * b2 +
        p[5] * s2 +
        p[6] * t2 +
        p[7] * monod_s +
        p[8] * gauss_t
    )
    
    # Add a small pH quadratic term and cross terms if we have room, but we're using all 10 params
    # Let's restructure: use 10 params more carefully
    
    # Actually, let me reconsider the parameter allocation:
    # p[0]: intercept
    # p[1]: b coefficient
    # p[2]: s coefficient
    # p[3]: temp coefficient
    # p[4]: pH coefficient
    # p[5]: b^2 coefficient
    # p[6]: s^2 coefficient
    # p[7]: Monod K_s (via log/exp)
    # p[8]: temp optimum offset (via tanh)
    # p[9]: temp width (via log/exp)
    
    # But we also want interaction terms. Let's use a different structure:
    # Combine quadratic surface with biological features more carefully
    
    # Let me redo this with a cleaner 10-parameter allocation:
    # p[0]: intercept
    # p[1]: b linear
    # p[2]: s linear
    # p[3]: temp linear
    # p[4]: pH linear
    # p[5]: b*s interaction
    # p[6]: b^2 coefficient
    # p[7]: Monod half-saturation (log scale)
    # p[8]: temp optimum (offset from 30)
    # p[9]: temp width (log scale)
    
    K_s = np.exp(p[7])
    T_opt = 30.0 + p[8]
    sigma_t = np.exp(p[9]) + 0.5
    
    monod_s = s / (K_s + s)
    gauss_t = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_t * sigma_t))
    
    pred = (
        p[0] +
        p[1] * b_c +
        p[2] * s_c +
        p[3] * t_c +
        p[4] * p_c +
        p[5] * bs +
        p[6] * b2 +
        p[7] * monod_s +
        p[8] * gauss_t
    )
    
    return pred
# EVOLVE-BLOCK-END
