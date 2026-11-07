"""E. coli growth rate model with rational (Padé-like) structure and smooth Gaussian bases."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline_start = time.time()
_DEADLINE = 140.0  # Safety margin before 150s wall-clock limit


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as a rational function (Padé-like) of smooth basis functions.
    
    Structure: prediction = (p0 * N1 + p1 * N2 + ... + p5 * N6) / (p6 * D1 + p7 * D2 + ... + p9 * D5)
    
    This rational form allows the model to capture saturation, inhibition, and complex
    non-linear interactions more efficiently than additive sums. The numerator encodes
    growth-promoting factors (substrate, temperature, pH) while the denominator encodes
    inhibitory factors (density, excess substrate, deviation from optima).
    
    Key advantages:
      - Smooth, differentiable everywhere (good for BFGS)
      - Rational functions can approximate complex curves with fewer parameters
      - Natural encoding of inhibition via denominator terms
      - Flexible interaction structure via numerator/denominator basis combinations
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological centers (based on E. coli physiology)
    T_center = 37.0    # Optimal temperature
    T_sigma = 4.0      # Temperature Gaussian width
    pH_center = 7.0    # Optimal pH
    pH_sigma = 1.2     # pH Gaussian width
    Ks = 3.0           # Substrate half-saturation constant
    Kd = 0.5           # Density inhibition constant
    
    # Deadline guard
    if time.time() - _deadline_start > _DEADLINE:
        return np.zeros_like(b)
    
    # Smooth Gaussian temperature response
    temp_gauss = np.exp(-((temp - T_center) ** 2) / (2.0 * T_sigma ** 2))
    
    # Smooth Gaussian pH response
    ph_gauss = np.exp(-((pH - pH_center) ** 2) / (2.0 * pH_sigma ** 2))
    
    # Monod substrate saturation
    monod_s = s / (Ks + s)
    
    # Density inhibition (smooth, monotone decreasing)
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # Numerator basis functions (growth-promoting terms)
    # N1: Full interaction of all factors
    N1 = monod_s * temp_gauss * ph_gauss * density_inhib
    # N2: Substrate-temperature interaction
    N2 = monod_s * temp_gauss
    # N3: Substrate-pH interaction
    N3 = monod_s * ph_gauss
    # N4: Temperature-pH interaction
    N4 = temp_gauss * ph_gauss
    # N5: Substrate only
    N5 = monod_s
    # N6: Constant term
    N6 = np.ones_like(b)
    
    # Denominator basis functions (inhibitory terms)
    # D1: Constant (baseline)
    D1 = np.ones_like(b)
    # D2: Density effect (stronger inhibition at high density)
    D2 = b / (1.0 + b)
    # D3: Substrate excess (inhibition at very high substrate)
    D3 = (s / (Ks + s)) ** 2
    # D4: Temperature deviation penalty
    D4 = 1.0 - temp_gauss
    # D5: pH deviation penalty
    D5 = 1.0 - ph_gauss
    
    # Build numerator
    numerator = (
        p0 * N1 +
        p1 * N2 +
        p2 * N3 +
        p3 * N4 +
        p4 * N5 +
        p5 * N6
    )
    
    # Build denominator (ensure it stays positive and bounded away from zero)
    denominator = (
        p6 * D1 +
        p7 * D2 +
        p8 * D3 +
        p9 * D4 +
        1.0 * D5  # Fixed coefficient for pH deviation to ensure stability
    )
    
    # Ensure denominator is positive and not too small to avoid division issues
    denominator = np.maximum(denominator, 0.1)
    
    # Rational prediction
    prediction = numerator / denominator
    
    return prediction
# EVOLVE-BLOCK-END
