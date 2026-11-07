"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics for substrate, 
    Gaussian responses for temperature and pH, and linear density dependence.
    
    Inputs:
    b: Population density
    s: Substrate concentration
    temp: Temperature
    pH: pH level
    
    Params (length 10):
    p[0]: Substrate max effect
    p[1]: Monod constant Ks
    p[2]: Substrate baseline offset
    p[3]: Optimal Temperature
    p[4]: Temperature width
    p[5]: Optimal pH
    p[6]: pH width
    p[7]: Density coefficient
    p[8]: Density offset
    p[9]: Global offset
    """
    p = params
    
    # Monod kinetics for substrate
    # Add small epsilon to prevent division by zero if p[1] becomes negative or zero
    ks = p[1] + 1e-6
    monod = (p[0] * s) / (ks + s) + p[2]
    
    # Gaussian response for temperature
    # Ensure width is non-zero
    w_t = p[4] + 1e-6
    temp_factor = np.exp(-((temp - p[3]) ** 2) / (2 * w_t ** 2))
    
    # Gaussian response for pH
    w_ph = p[6] + 1e-6
    ph_factor = np.exp(-((pH - p[5]) ** 2) / (2 * w_ph ** 2))
    
    # Linear density dependence
    density_factor = p[7] * b + p[8]
    
    # Combine
    return monod * temp_factor * ph_factor * density_factor + p[9]
# EVOLVE-BLOCK-END
