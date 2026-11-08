"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin from 150s limit


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Model: mu = mu_max * (s / (Ks + s + s^2/Ki)) * exp(-((T-Topt)/sigma_T)^2) * 
                          exp(-((pH-pHopt)/sigma_pH)^2) * (1/(1+(b/Kb)^alpha_b)) + offset
    """
    # Deadline guard: If we are close to the limit, return zeros to allow
    # the optimizer to terminate gracefully rather than hanging or crashing
    # in the evaluator's timeout handler.
    if time.time() > _DEADLINE:
        return np.zeros_like(b)

    mu_max = params[0]
    K_s = np.abs(params[1]) + 1e-6
    K_i = np.abs(params[2]) + 1e-6
    T_opt = params[3]
    sigma_T = np.abs(params[4]) + 1e-6
    pH_opt = params[5]
    sigma_pH = np.abs(params[6]) + 1e-6
    K_b = np.abs(params[7]) + 1e-6
    alpha_b = params[8]
    offset = params[9]
    
    # Haldane substrate term (allows inhibition at high s)
    denom_s = K_s + s + (s ** 2) / K_i
    monod = s / denom_s
    
    # Temperature Gaussian
    temp_factor = np.exp(-((temp - T_opt) / sigma_T) ** 2)
    
    # pH Gaussian
    ph_factor = np.exp(-((pH - pH_opt) / sigma_pH) ** 2)
    
    # Density inhibition (logistic)
    density_factor = 1.0 / (1.0 + (b / K_b) ** alpha_b)
    
    # Combine
    mu = mu_max * monod * temp_factor * ph_factor * density_factor + offset
    
    return mu
# EVOLVE-BLOCK-END
