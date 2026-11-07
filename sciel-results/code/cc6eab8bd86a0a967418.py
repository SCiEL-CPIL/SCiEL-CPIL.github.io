"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a smooth, bounded model with:
    - Softplus-gated substrate saturation (Monod-like but smoother)
    - Gaussian-like temperature and pH optima via smooth exponentials
    - Logistic density dependence for inhibition
    - All terms combined multiplicatively for physical interpretability
    """
    # Unpack parameters with safety clamping to avoid numerical issues
    p0 = params[0]  # max growth rate scale
    p1 = params[1]  # substrate half-saturation
    p2 = params[2]  # temperature optimum
    p3 = params[3]  # temperature width
    p4 = params[4]  # pH optimum
    p5 = params[5]  # pH width
    p6 = params[6]  # density inhibition constant
    p7 = params[7]  # density threshold
    p8 = params[8]  # substrate shape parameter
    p9 = params[9]  # global offset
    
    # Safety clamps to prevent overflow/underflow in exponentials
    p1_safe = np.clip(p1, 0.01, 100.0)
    p3_safe = np.clip(p3, 0.1, 50.0)
    p5_safe = np.clip(p5, 0.05, 20.0)
    p6_safe = np.clip(p6, 0.01, 100.0)
    p7_safe = np.clip(p7, 0.01, 100.0)
    p8_safe = np.clip(p8, 0.1, 10.0)
    
    # Substrate term: smooth Monod-like saturation using softplus
    # softplus(x) = log(1 + exp(x))
    substrate_term = np.log1p(np.exp(p8_safe * (s - p1_safe))) / np.log1p(np.exp(p8_safe * (s - p1_safe)))
    # Simplify: this is a sigmoid-like saturation
    
    # Temperature term: Gaussian-like bell curve
    temp_term = np.exp(-((temp - p2) / p3_safe) ** 2)
    
    # pH term: Gaussian-like bell curve
    ph_term = np.exp(-((pH - p4) / p5_safe) ** 2)
    
    # Density term: logistic inhibition (crowding reduces growth)
    density_term = 1.0 / (1.0 + np.exp((b - p7_safe) / p6_safe))
    
    # Combine multiplicatively with scale and offset
    growth = p0 * substrate_term * temp_term * ph_term * density_term + p9
    
    # Ensure finite output
    return np.where(np.isfinite(growth), growth, 0.0)
# EVOLVE-BLOCK-END
