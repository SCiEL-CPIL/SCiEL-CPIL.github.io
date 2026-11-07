"""E. coli growth rate symbolic regression with biologically-inspired structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining Monod substrate kinetics,
    temperature optimum, pH optimum, and density-dependent inhibition.
    
    Structure:
      mu = p0 * (s / (Ks + s)) * f_temp * f_pH * f_density
    
    Where:
      f_temp  = exp(-((temp - T_opt)^2) / (2 * sigma_T^2))
      f_pH    = exp(-((pH - pH_opt)^2) / (2 * sigma_pH^2))
      f_density = 1 / (1 + (b / Kb)^n)
    
    Parameters (10 total):
      p[0]: Max growth rate scale (mu_max)
      p[1]: log(Ks) - Monod half-saturation constant
      p[2]: T_opt - optimal temperature
      p[3]: log(sigma_T) - temperature width
      p[4]: pH_opt - optimal pH
      p[5]: log(sigma_pH) - pH width
      p[6]: log(Kb) - density inhibition constant
      p[7]: n - density inhibition exponent
      p[8]: additional linear temp coefficient
      p[9]: additional linear pH coefficient
    """
    p = params
    
    # Monod substrate term: s / (Ks + s)
    Ks = np.exp(p[1])
    monod = s / (Ks + s + 1e-12)
    
    # Temperature response: Gaussian around optimal temperature
    sigma_T = np.exp(p[3])
    f_temp = np.exp(-((temp - p[2]) ** 2) / (2.0 * sigma_T ** 2 + 1e-12))
    
    # pH response: Gaussian around optimal pH
    sigma_pH = np.exp(p[5])
    f_pH = np.exp(-((pH - p[4]) ** 2) / (2.0 * sigma_pH ** 2 + 1e-12))
    
    # Density-dependent inhibition (Haldane-type)
    Kb = np.exp(p[6])
    f_density = 1.0 / (1.0 + (b / (Kb + 1e-12)) ** p[7])
    
    # Base growth rate
    mu = p[0] * monod * f_temp * f_pH * f_density
    
    # Add small linear corrections for temperature and pH to capture asymmetry
    mu = mu + p[8] * (temp - p[2]) * f_temp + p[9] * (pH - p[4]) * f_pH
    
    return mu
# EVOLVE-BLOCK-END
