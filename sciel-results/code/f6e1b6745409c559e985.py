"""E. coli growth rate symbolic regression with Monod kinetics and temperature/pH effects."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    params: array of length 10
      p[0]: max growth rate multiplier
      p[1]: log(Ks) - log half-saturation constant for substrate
      p[2]: log(Kb) - log half-saturation constant for biomass (inhibition)
      p[3]: temperature optimum (shift)
      p[4]: temperature sensitivity
      p[5]: pH optimum (shift)
      p[6]: pH sensitivity
      p[7]: Hill coefficient for substrate
      p[8]: biomass inhibition coefficient
      p[9]: bias
    """
    p = params
    
    # Extract parameters with safe transformations
    max_rate = np.exp(p[0])  # Ensure positive max rate
    Ks = np.exp(p[1])        # Ensure positive half-saturation for substrate
    Kb = np.exp(p[2])        # Ensure positive half-saturation for biomass
    temp_opt = 37.0 + p[3]   # Temperature optimum around 37C with shift
    temp_sens = np.exp(p[4]) # Temperature sensitivity (positive)
    ph_opt = 7.0 + p[5]      # pH optimum around 7 with shift
    ph_sens = np.exp(p[6])   # pH sensitivity (positive)
    hill = 1.0 + np.abs(p[7]) # Hill coefficient >= 1
    bi_inhib = np.exp(p[8])   # Biomass inhibition coefficient
    bias = p[9]
    
    # Monod term for substrate: s^n / (Ks^n + s^n)
    s_safe = np.maximum(s, 1e-10)
    monod_s = (s_safe ** hill) / (Ks ** hill + s_safe ** hill)
    
    # Temperature effect: Gaussian around optimum
    temp_diff = temp - temp_opt
    temp_effect = np.exp(-0.5 * (temp_diff / temp_sens) ** 2)
    
    # pH effect: Gaussian around optimum
    ph_diff = pH - ph_opt
    ph_effect = np.exp(-0.5 * (ph_diff / ph_sens) ** 2)
    
    # Biomass inhibition: 1 / (1 + (b/Kb)^2) - prevents overpopulation
    b_safe = np.maximum(b, 1e-10)
    biomass_inhib = 1.0 / (1.0 + (b_safe / Kb) ** 2)
    
    # Combine all terms
    growth = max_rate * monod_s * temp_effect * ph_effect * biomass_inhib + bias
    
    return growth
# EVOLVE-BLOCK-END
