import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with Fourier basis for temperature and pH.
    
    Uses fixed-frequency sine/cosine terms to capture asymmetric or non-Gaussian
    response curves for temperature and pH. Substrate and density terms remain
    mechanistic (Monod-like).
    
    Parameters (10 total):
    p0: Max growth rate scale
    p1: log(Ks) - substrate half-saturation
    p2: Substrate scaling factor
    p3: Temperature linear coefficient
    p4: Temperature quadratic coefficient (for asymmetry)
    p5: pH linear coefficient
    p6: pH quadratic coefficient (for asymmetry)
    p7: log(Kb) - density inhibition constant
    p8: logit(inhibition_strength) - mapped to (0,1) via sigmoid
    p9: Bias term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Smooth positive mappings
    Ks = np.exp(p1) + 0.01
    Kb = np.exp(p7) + 0.01
    # Sigmoid for inhibition strength in (0, 1)
    inhib_strength = 1.0 / (1.0 + np.exp(-p8))
    
    # Substrate term: Monod
    S_term = (s / (Ks + s)) * p2
    
    # Temperature term: Fourier basis with fixed frequencies
    # Use frequencies that capture the main variation in [1, 50] range
    # omega_T = 2*pi/10 ≈ 0.628, so period ~10°C
    omega_T = 2.0 * np.pi / 10.0
    T_term = 1.0 + p3 * np.sin(omega_T * temp) + p4 * np.cos(omega_T * temp)
    
    # pH term: Fourier basis with fixed frequencies
    # pH range is typically ~4-10, use period ~3
    omega_pH = 2.0 * np.pi / 3.0
    pH_term = 1.0 + p5 * np.sin(omega_pH * pH) + p6 * np.cos(omega_pH * pH)
    
    # Ensure positivity of environmental terms by using exp
    T_term = np.exp(T_term)
    pH_term = np.exp(pH_term)
    
    # Density inhibition term
    B_term = 1.0 - inhib_strength * (b / (Kb + b))
    
    # Combine
    growth = p0 * S_term * T_term * pH_term * B_term + p9
    
    return growth
