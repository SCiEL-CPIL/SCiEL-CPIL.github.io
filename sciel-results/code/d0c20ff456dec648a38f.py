"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Uses a composite model:
    - Monod kinetics for substrate (s) with a log-parameterized Ks for stability.
    - Skewed Gaussian response for temperature (temp).
    - Skewed Gaussian response for pH.
    - Exponential inhibition for biomass (b).
    - Constant offset.
    
    Parameters (10 total):
    p[0]: Amplitude (max growth rate scale)
    p[1]: Log of Ks (Monod constant)
    p[2]: Temp Gaussian width coefficient (inverse variance)
    p[3]: Temp optimal value (center)
    p[4]: pH Gaussian width coefficient (inverse variance)
    p[5]: pH optimal value (center)
    p[6]: Biomass inhibition coefficient
    p[7]: Offset
    p[8]: Temp linear skew coefficient
    p[9]: pH linear skew coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Monod term: s / (Ks + s), with Ks = exp(p1) to ensure positivity
    monod = s / (np.exp(p1) + s)
    
    # Temperature term: skewed Gaussian
    # exp(-p2 * (temp - p3)^2 + p8 * temp)
    temp_term = np.exp(-p2 * (temp - p3) ** 2 + p8 * temp)
    
    # pH term: skewed Gaussian
    # exp(-p4 * (pH - p5)^2 + p9 * pH)
    ph_term = np.exp(-p4 * (pH - p5) ** 2 + p9 * pH)
    
    # Biomass inhibition: exp(-p6 * b)
    bio_term = np.exp(-p6 * b)
    
    # Combine
    return p0 * monod * temp_term * ph_term * bio_term + p7
# EVOLVE-BLOCK-END
