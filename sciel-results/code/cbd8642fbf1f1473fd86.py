"""E. coli growth rate model: Monod substrate x Gaussian temp/pH with flexibility."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with biologically-motivated multiplicative structure.
    
    mu = p0 * (s / (p1 + s)) * exp(-0.5 * ((temp - p2) / p3)^2) * exp(-0.5 * ((pH - p4) / p5)^2)
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate concentration (range ~[1, 5])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[2, 12])
    
    Parameters (10):
    p0: Maximum growth rate
    p1: Monod half-saturation constant for substrate
    p2: Optimal temperature
    p3: Temperature breadth
    p4: Optimal pH
    p5: pH breadth
    p6: Additional linear temp term coefficient
    p7: Additional linear pH term coefficient
    p8: Additional quadratic temp term coefficient
    p9: Additional quadratic pH term coefficient
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Monod term for substrate
    monod = s / (p1 + s)
    
    # Gaussian term for temperature
    temp_gauss = np.exp(-0.5 * ((temp - p2) / p3) ** 2)
    
    # Gaussian term for pH
    ph_gauss = np.exp(-0.5 * ((pH - p4) / p5) ** 2)
    
    # Additional flexibility terms
    temp_linear = p6 * (temp - p2)
    ph_linear = p7 * (pH - p4)
    temp_quad = p8 * ((temp - p2) / p3) ** 2
    ph_quad = p9 * ((pH - p4) / p5) ** 2
    
    # Combine multiplicative core with additive corrections
    pred = p0 * monod * temp_gauss * ph_gauss * (1.0 + temp_linear + ph_linear + temp_quad + ph_quad)
    
    # Ensure the output is finite to prevent crashing the evaluator
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
