"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a biophysical multiplicative structure.
    
    This approach uses Monod kinetics for substrate, Gaussian optima for temperature
    and pH, and interaction terms for population density and environmental coupling.
    Inputs are scaled to [0, 1] ranges to ensure numerical stability for BFGS
    starting from initial parameters of 1.
    
    Parameters:
    b: Population density
    s: Substrate concentration
    temp: Temperature
    pH: pH level
    params: Array of 10 fitted constants.
        p[0]: Global scale factor
        p[1]: Substrate half-saturation constant (Ks)
        p[2]: Optimal temperature (scaled)
        p[3]: Temperature width (scaled)
        p[4]: Optimal pH (scaled)
        p[5]: pH width (scaled)
        p[6]: Population density coefficient
        p[7]: Temperature-pH interaction coefficient
        p[8]: Substrate-population interaction coefficient
        p[9]: Bias term
    """
    p = params
    
    # Scale inputs to approximately [0, 1] to match initial param values
    # Ranges: b[1,5], s[1,5], temp[1,40], pH[2,12]
    bs = b / 5.0
    ss = s / 5.0
    ts = temp / 40.0
    hs = pH / 12.0
    
    # Substrate Kinetics (Monod)
    # Ensure Ks is positive to avoid division by zero or negative denominators
    k_s = np.abs(p[1]) + 1e-6
    monod = ss / (k_s + ss)
    
    # Temperature Optimum (Gaussian)
    # Width must be positive
    t_width = np.abs(p[3]) + 1e-6
    t_opt = p[2]
    temp_factor = np.exp(-((ts - t_opt) / t_width) ** 2)
    
    # pH Optimum (Gaussian)
    h_width = np.abs(p[5]) + 1e-6
    h_opt = p[4]
    ph_factor = np.exp(-((hs - h_opt) / h_width) ** 2)
    
    # Interaction Terms
    # These allow the model to deviate from pure multiplicativity
    pop_factor = 1.0 + p[6] * bs
    th_interaction = 1.0 + p[7] * ts * hs
    sb_interaction = 1.0 + p[8] * ss * bs
    
    # Combine
    y = p[0] * monod * temp_factor * ph_factor * pop_factor * th_interaction * sb_interaction + p[9]
    
    return y
# EVOLVE-BLOCK-END
