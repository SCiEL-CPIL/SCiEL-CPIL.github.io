"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a multiplicative model combining:
    - Monod kinetics for substrate saturation
    - Cauchy (Lorentzian) bell curves for temperature and pH optima
    - Quadratic density dependence for inhibition/boosting
    
    This structure captures the non-linear, saturating nature of biological
    growth better than a simple linear combination.
    """
    # Unpack parameters
    c0 = params[0]
    c1 = params[1]
    Ks = params[2]
    muT = params[3]
    sigT = params[4]
    mupH = params[5]
    sigpH = params[6]
    alpha_b = params[7]
    beta_b = params[8]
    h = params[9]
    
    # Safety clamps to prevent numerical instability and division by zero
    # We use abs to allow the optimizer to explore negative values which might
    # correspond to sign flips in the numerator/denominator, but ensure magnitude
    # is non-zero.
    Ks_safe = np.maximum(np.abs(Ks), 1e-6)
    sigT_safe = np.maximum(np.abs(sigT), 1e-3)
    sigpH_safe = np.maximum(np.abs(sigpH), 1e-3)
    h_safe = np.maximum(np.abs(h), 1e-3)
    
    # 1. Substrate: Hill / Monod Kinetics
    # ratio_s = s / Ks
    # term_s = ratio_s^h / (1 + ratio_s^h)
    # If h=1, this is Monod: s/(Ks+s). 
    # Using a slightly more stable formulation: 1 / (1 + (Ks/s)^h) is dangerous if s=0.
    # The form (s/Ks)^h / (1 + (s/Ks)^h) is safe if Ks > 0.
    ratio_s = s / Ks_safe
    # Clip ratio_s to avoid overflow if s is very large and Ks very small, 
    # though inputs are bounded [1, 40] approx, so this is mostly safe.
    # However, if h is large, ratio^h can overflow. 
    # Let's use a stable sigmoid-like form: 1 / (1 + (Ks_safe/s)**h) is bad at s=0.
    # The original form is fine for s>0. Inputs s are in [1, 40].
    term_s = (ratio_s ** h_safe) / (1.0 + ratio_s ** h_safe)
    
    # 2. Temperature: Cauchy (Lorentzian) Bell Curve
    # More robust gradients than Gaussian, especially in tails.
    delta_t = (temp - muT) / sigT_safe
    term_t = 1.0 / (1.0 + delta_t ** 2)
    
    # 3. pH: Cauchy (Lorentzian) Bell Curve
    delta_p = (pH - mupH) / sigpH_safe
    term_p = 1.0 / (1.0 + delta_p ** 2)
    
    # 4. Density: Quadratic Dampening/Boosting
    # E. coli often exhibits density-dependent effects (quorum sensing, waste accumulation).
    # term_b = 1 + alpha*b + beta*b^2
    term_b = 1.0 + alpha_b * b + beta_b * (b ** 2)
    
    # Combine
    # c1 scales the main multiplicative effect. c0 is a baseline offset.
    # The product of factors ensures that if any condition is sub-optimal (factor -> 0), growth -> 0.
    y = c0 + c1 * term_s * term_t * term_p * term_b
    
    return y
# EVOLVE-BLOCK-END
