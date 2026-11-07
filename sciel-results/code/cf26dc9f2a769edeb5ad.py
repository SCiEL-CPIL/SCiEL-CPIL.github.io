"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a biologically-motivated basis expansion.
    
    The model is linear in params, allowing BFGS to converge to the global optimum.
    Basis functions:
    1. Bias
    2. Linear density (b)
    3. Linear substrate (s)
    4. Linear temperature (temp)
    5. Linear pH (pH)
    6. Monod substrate saturation: s / (1 + s)
    7. Gaussian temperature optimum: exp(-(temp - 37)^2 / 50)
    8. Gaussian pH optimum: exp(-(pH - 7)^2 / 8)
    9. Density inhibition: 1 / (1 + b)
    10. Substrate-Temperature interaction: (s / (1 + s)) * exp(-(temp - 37)^2 / 50)
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Pre-compute basis functions
    # 1. Bias
    bias = np.ones_like(b)
    
    # 2. Linear terms
    lin_b = b
    lin_s = s
    lin_temp = temp
    lin_ph = pH
    
    # 3. Non-linear biological terms
    # Monod substrate saturation (Ks = 1.0)
    monod_s = s / (1.0 + s)
    
    # Temperature response (Optimum ~37C, Width ~5C)
    temp_gauss = np.exp(-((temp - 37.0) ** 2) / 50.0)
    
    # pH response (Optimum ~7, Width ~2)
    ph_gauss = np.exp(-((pH - 7.0) ** 2) / 8.0)
    
    # Density inhibition (Contact inhibition)
    dens_inhib = 1.0 / (1.0 + b)
    
    # Interaction: Substrate availability modulates temperature response
    inter_s_temp = monod_s * temp_gauss
    
    # Combine linearly
    return (p0 * bias + 
            p1 * lin_b + 
            p2 * lin_s + 
            p3 * lin_temp + 
            p4 * lin_ph + 
            p5 * monod_s + 
            p6 * temp_gauss + 
            p7 * ph_gauss + 
            p8 * dens_inhib + 
            p9 * inter_s_temp)
# EVOLVE-BLOCK-END
