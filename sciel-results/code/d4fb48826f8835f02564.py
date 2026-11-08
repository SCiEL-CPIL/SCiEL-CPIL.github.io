"""E. coli growth rate model with Monod kinetics, Gaussian temp/pH optima, and density inhibition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Model: mu_max * (S/(Ks+S)) * exp(-((T-T_opt)^2)/(2*sigma_T^2)) * exp(-((pH-pH_opt)^2)/(2*sigma_pH^2)) * 1/(1+Kb*b)
    
    Parameters:
    b: population density
    s: substrate concentration
    temp: temperature
    pH: pH
    params: 10 parameters
        p[0]: mu_max
        p[1]: Ks
        p[2]: T_opt
        p[3]: sigma_T
        p[4]: pH_opt
        p[5]: sigma_pH
        p[6]: Kb
        p[7]: offset
        p[8]: scale_s
        p[9]: scale_b
    """
    mu_max = params[0]
    Ks = np.abs(params[1]) + 1e-6
    T_opt = params[2]
    sigma_T = np.abs(params[3]) + 1e-6
    pH_opt = params[4]
    sigma_pH = np.abs(params[5]) + 1e-6
    Kb = np.abs(params[6]) + 1e-6
    offset = params[7]
    scale_s = np.abs(params[8]) + 1e-6
    scale_b = np.abs(params[9]) + 1e-6
    
    # Monod kinetics for substrate
    monod = (s * scale_s) / (Ks + s * scale_s)
    
    # Gaussian temperature optimum
    temp_factor = np.exp(-((temp - T_opt) ** 2) / (2 * sigma_T ** 2))
    
    # Gaussian pH optimum
    ph_factor = np.exp(-((pH - pH_opt) ** 2) / (2 * sigma_pH ** 2))
    
    # Density inhibition
    density_factor = 1.0 / (1.0 + Kb * b * scale_b)
    
    return mu_max * monod * temp_factor * ph_factor * density_factor + offset
# EVOLVE-BLOCK-END
