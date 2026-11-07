"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Uses a composite model:
    - Monod kinetics for substrate (s)
    - Gaussian response for temperature (temp) with a linear skew term
    - Gaussian response for pH with a linear skew term
    - Exponential inhibition for biomass (b)
    - Constant offset
    
    Parameters (10 total):
    p[0]: Amplitude
    p[1]: Ks (Monod constant)
    p[2]: Temp Gaussian width coefficient
    p[3]: Temp optimal value
    p[4]: pH Gaussian width coefficient
    p[5]: pH optimal value
    p[6]: Biomass inhibition coefficient
    p[7]: Offset
    p[8]: Temp linear skew coefficient
    p[9]: pH linear skew coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Monod term for substrate
    # Avoid division by zero if p1 is close to 0, though BFGS should handle it
    # We use a small epsilon for safety
    monod = s / (p1 + s + 1e-9)
    
    # Temperature term: Gaussian + linear skew
    # exp(-p2 * (temp - p3)^2 + p8 * temp)
    temp_term = np.exp(-p2 * (temp - p3)**2 + p8 * temp)
    
    # pH term: Gaussian + linear skew
    # exp(-p4 * (pH - p5)^2 + p9 * pH)
    ph_term = np.exp(-p4 * (pH - p5)**2 + p9 * pH)
    
    # Biomass inhibition term
    b_term = np.exp(-p6 * b)
    
    # Combine
    return p0 * monod * temp_term * ph_term * b_term + p7
# EVOLVE-BLOCK-END
