"""E. coli growth rate model with Monod kinetics, Gaussian temp/pH optima, and density inhibition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Model: mu_max * (S/(Ks+S)) * exp(-((T-T_opt)^2)/(2*sigma_T^2)) * exp(-((pH-pH_opt)^2)/(2*sigma_pH^2)) * 1/(1+Kb*(b-b_min))
    
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
        p[7]: b_min
        p[8]: offset
        p[9]: scale
    """
    mu_max = params[0]
    Ks = params[1]
    T_opt = params[2]
    sigma_T = params[3]
    pH_opt = params[4]
    sigma_pH = params[5]
    Kb = params[6]
    b_min = params[7]
    offset = params[8]
    scale = params[9]
    
    # Monod substrate term
    monod = s / (Ks + s)
    
    # Gaussian temperature term
    temp_term = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T ** 2))
    
    # Gaussian pH term
    ph_term = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density inhibition term (smooth logistic via softplus)
    # Use softplus approximation: log(1+exp(x)) ≈ x for x>0, ~0 for x<0
    x = Kb * (b - b_min)
    # Softplus: log(1+exp(x)) with numerical stability
    softplus_x = np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 50))))
    density_term = np.exp(-softplus_x)
    
    # Combine
    pred = mu_max * monod * temp_term * ph_term * density_term * scale + offset
    
    return pred
# EVOLVE-BLOCK-END
