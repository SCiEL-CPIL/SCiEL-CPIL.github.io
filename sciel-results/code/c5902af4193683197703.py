"""E. coli growth rate model with dual density term and gamma cross-term."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining:
    - Monod-like substrate saturation
    - Gaussian temperature and pH optima
    - Dual density response: activation at low density, inhibition at high density
    - Gamma cross-term for substrate-temperature-density interaction
    
    params (10 constants):
        p0: max growth rate scale
        p1: substrate half-saturation constant
        p2: temperature optimum
        p3: temperature width
        p4: pH optimum
        p5: pH width
        p6: density activation threshold
        p7: density inhibition threshold
        p8: gamma cross-term coefficient
        p9: baseline offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Substrate saturation (Monod-like, smooth)
    # Use a smooth approximation to avoid division by zero
    s_safe = np.maximum(s, 1e-8)
    monod = s_safe / (np.abs(p1) + s_safe)
    
    # Temperature response: Gaussian centered at p2 with width p3
    temp_diff = temp - p2
    temp_width = np.maximum(np.abs(p3), 1e-6)
    temp_response = np.exp(-0.5 * (temp_diff / temp_width) ** 2)
    
    # pH response: Gaussian centered at p4 with width p5
    ph_diff = pH - p4
    ph_width = np.maximum(np.abs(p5), 1e-6)
    ph_response = np.exp(-0.5 * (ph_diff / ph_width) ** 2)
    
    # Dual density response:
    # - Activation term: logistic that increases with b, threshold p6
    # - Inhibition term: logistic that decreases with b, threshold p7
    b_safe = np.maximum(b, 1e-8)
    activation = 1.0 / (1.0 + np.exp(-b_safe / np.maximum(np.abs(p6), 1e-6)))
    inhibition = 1.0 / (1.0 + np.exp((b_safe - np.abs(p7)) / np.maximum(np.abs(p7), 1e-6)))
    
    # Combine density effects multiplicatively
    density_effect = activation * inhibition
    
    # Gamma cross-term: interaction between substrate, temperature, and density
    # This allows the model to capture non-linear density effects that depend on
    # substrate availability and temperature
    gamma_term = 1.0 + p8 * monod * temp_response * (b_safe / (np.abs(p6) + b_safe))
    
    # Final prediction
    prediction = p0 * monod * temp_response * ph_response * density_effect * gamma_term + p9
    
    # Ensure finite output
    return np.where(np.isfinite(prediction), prediction, 0.0)
# EVOLVE-BLOCK-END
