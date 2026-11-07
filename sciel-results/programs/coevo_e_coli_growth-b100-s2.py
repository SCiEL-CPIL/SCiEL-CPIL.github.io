import numpy as np

def _softplus(x):
    """
    Numerically stable softplus function: log(1 + exp(x)).
    Smooth approximation to max(x, 0).
    """
    # Avoid overflow in exp(x) for large x
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Biologically-motivated model for E. coli growth rate.
    Structure: mu_max * Monod(s) * Gaussian(T) * Gaussian(pH) * Interaction(b, s)
    
    Parameters:
    0: mu_max (max growth rate scale)
    1: log_Ks (log of Monod half-saturation constant)
    2: T_opt (optimal temperature, normalized)
    3: log_sigT (log of temperature response width)
    4: pH_opt (optimal pH, normalized)
    5: log_sigpH (log of pH response width)
    6: c_b (coefficient for biomass density)
    7: c_s (coefficient for substrate concentration)
    8: c_bs (coefficient for biomass-substrate interaction)
    9: bias (baseline biomass effect)
    """
    # Unpack parameters
    mu_max = params[0]
    log_Ks = params[1]
    T_opt = params[2]
    log_sigT = params[3]
    pH_opt = params[4]
    log_sigpH = params[5]
    c_b = params[6]
    c_s = params[7]
    c_bs = params[8]
    bias = params[9]
    
    # Normalize inputs to [0, 1] based on known ranges
    # b: [1, 4.998] -> approx [1, 5]
    # s: [1, 39.98] -> approx [1, 40]
    # temp: [1, 12]
    # pH: [1, 4.999] -> approx [1, 5]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Ensure positive widths and Ks
    Ks = _softplus(log_Ks) + 1e-6
    sigT = _softplus(log_sigT) + 1e-6
    sigpH = _softplus(log_sigpH) + 1e-6
    
    # Monod kinetics for substrate
    # s_norm is in [0, 1]. Ks is positive.
    monod_s = s_norm / (Ks + s_norm)
    
    # Gaussian response for temperature
    # temp_norm is in [0, 1]. T_opt is in same scale.
    temp_resp = np.exp(-((temp_norm - T_opt) ** 2) / (2.0 * sigT ** 2))
    
    # Gaussian response for pH
    # ph_norm is in [0, 1]. pH_opt is in same scale.
    ph_resp = np.exp(-((ph_norm - pH_opt) ** 2) / (2.0 * sigpH ** 2))
    
    # Interaction term for biomass and substrate
    # This term modulates the growth rate based on population density and substrate availability
    # It is a smooth polynomial in normalized inputs
    b_effect = bias + c_b * b_norm + c_s * s_norm + c_bs * b_norm * s_norm
    
    # Combine components
    mu = mu_max * monod_s * temp_resp * ph_resp * b_effect
    
    return mu
