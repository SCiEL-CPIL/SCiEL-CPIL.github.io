import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0

    p = params

    # Structural change: Use a mechanistic model with Monod kinetics for substrate,
    # Gaussian bell curves for temperature and pH, and a logistic-like term for biomass.
    # This is a different modelling family (mechanistic vs polynomial) and may better
    # capture the underlying biology of E. coli growth.
    
    # Substrate: Monod-like saturation
    # p[0] = half-saturation constant (k_s)
    k_s = np.abs(p[0]) + 1e-6
    term_s = s_norm / (s_norm + k_s)
    
    # Temperature: Gaussian bell curve
    # p[1] = optimal temperature (center in normalized scale)
    # p[2] = width parameter
    t_center = p[1]
    t_width = np.abs(p[2]) + 1e-6
    term_t = np.exp(-((t_norm - t_center) ** 2) / (2 * t_width ** 2))
    
    # pH: Gaussian bell curve
    # p[3] = optimal pH (center in normalized scale)
    # p[4] = width parameter
    ph_center = p[3]
    ph_width = np.abs(p[4]) + 1e-6
    term_ph = np.exp(-((ph_norm - ph_center) ** 2) / (2 * ph_width ** 2))
    
    # Biomass: Logistic-like inhibition
    # p[5] = inhibition strength
    # term_b = 1 / (1 + p[5] * b_norm)
    # But we want it to be smooth and differentiable.
    # Use a sigmoid-like function: 1 / (1 + exp(p[5] * (b_norm - 0.5)))
    # This captures the idea that growth is high at low biomass and decreases at high biomass.
    # However, this might be too strong. Let's use a simpler polynomial-like term.
    # Actually, let's stick with a simple rational function for biomass inhibition.
    # term_b = 1 / (1 + p[5] * b_norm)
    term_b = 1.0 / (1.0 + np.abs(p[5]) * b_norm)
    
    # Combine terms with multiplicative and additive structure
    # The growth rate is typically proportional to the product of environmental factors.
    # Let's use: y = p[6] * term_s * term_t * term_ph * term_b + p[7] * term_s * term_t + p[8] * term_t + p[9]
    # This allows for interactions between substrate and temperature, and temperature alone.
    
    y = (p[6] * term_s * term_t * term_ph * term_b +
         p[7] * term_s * term_t +
         p[8] * term_t +
         p[9])

    return y


