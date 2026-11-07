"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _softplus(x, beta=1.0):
    """Smooth approximation to max(x, 0), differentiable everywhere."""
    return np.log1p(np.exp(-beta * np.abs(x))) / beta + np.maximum(x, 0.0)


def _softabs(x, beta=1.0):
    """Smooth approximation to |x|, differentiable everywhere."""
    return x * np.tanh(beta * x) / beta + _softplus(x, beta)


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Biologically-motivated model for E. coli growth rate:
    - Monod kinetics for substrate (smooth positive Ks via softplus)
    - Gaussian responses for temperature and pH (smooth positive sigmas via softplus)
    - Smooth biomass density effect with T-pH interaction
    """
    # Unpack 10 parameters
    mu_max = params[0]
    log_Ks = params[1]      # log of Monod half-saturation
    T_opt = params[2]
    log_sigT = params[3]    # log of temperature width
    pH_opt = params[4]
    log_sigpH = params[5]   # log of pH width
    c_b = params[6]
    c_bs = params[7]
    c_tp = params[8]        # T-pH interaction coefficient
    bias = params[9]
    
    # Normalize inputs to roughly [0, 1] range
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Smooth positive parameters
    Ks = _softplus(log_Ks) + 1e-6
    sigT = _softplus(log_sigT) + 1e-6
    sigpH = _softplus(log_sigpH) + 1e-6
    
    # Monod kinetics for substrate: s / (Ks + s)
    monod_s = s_norm / (Ks + s_norm)
    
    # Gaussian temperature response
    temp_resp = np.exp(-((temp_norm - T_opt) ** 2) / (2.0 * sigT ** 2))
    
    # Gaussian pH response
    ph_resp = np.exp(-((ph_norm - pH_opt) ** 2) / (2.0 * sigpH ** 2))
    
    # Smooth biomass effect with T-pH interaction
    b_effect = bias + c_b * b_norm + c_bs * b_norm * s_norm + c_tp * temp_norm * ph_norm
    
    # Combine all components
    mu = mu_max * monod_s * temp_resp * ph_resp * b_effect
    
    return mu
# EVOLVE-BLOCK-END
