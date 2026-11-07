"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate with dual density response and cross-term:
    - Monod kinetics for substrate
    - Gaussian temperature and pH responses
    - Dual density term: inhibition * activation to capture non-linear density effects
    - Gamma cross-term: multiplicative interaction between substrate, temp, and density
    """
    # Unpack 10 parameters
    mu_max = params[0]      # Maximum growth rate
    Ks = params[1]          # Monod half-saturation constant
    T_opt = params[2]       # Optimal temperature
    sigT = params[3]        # Temperature width
    pH_opt = params[4]      # Optimal pH
    sigpH = params[5]       # pH width
    b_inh = params[6]       # Inhibition density scale
    b_act = params[7]       # Activation density scale
    offset = params[8]      # Baseline offset
    gamma = params[9]       # Cross-term coefficient
    
    # Safety
    Ks_safe = np.abs(Ks) + 1e-10
    sigT_safe = np.abs(sigT) + 1e-10
    sigpH_safe = np.abs(sigpH) + 1e-10
    b_inh_safe = np.abs(b_inh) + 1e-10
    b_act_safe = np.abs(b_act) + 1e-10
    
    # Monod substrate term
    monod = s / (Ks_safe + s)
    
    # Gaussian temperature response
    temp_resp = np.exp(-((temp - T_opt) ** 2) / (2 * sigT_safe ** 2))
    
    # Gaussian pH response
    ph_resp = np.exp(-((pH - pH_opt) ** 2) / (2 * sigpH_safe ** 2))
    
    # Dual density: inhibition decreases growth at high density,
    # activation increases growth at moderate density
    inhib = np.exp(-b / b_inh_safe)
    activ = b / (b_act_safe + b)
    density_dual = inhib * activ
    
    # Base multiplicative model
    base = mu_max * monod * temp_resp * ph_resp * density_dual
    
    # Cross-term: captures interaction between substrate, temperature, and density
    # This adds a second pathway that can't be expressed as pure multiplication
    cross = gamma * monod * temp_resp * (b / (b_act_safe + b))
    
    mu = base + cross + offset
    
    return mu
# EVOLVE-BLOCK-END
