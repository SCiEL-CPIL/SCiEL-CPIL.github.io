"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series approximation with 10 parameters.
    Maps input domain [0.1, 6] to [0, 1] and uses harmonics.
    params[0]: DC term (constant)
    params[1:9]: Coefficients for harmonics 1-4 (cos, sin pairs)
    params[9]: Coefficient for harmonic 5 (cos only, to fit 10 params)
    """
    # Normalize x to [0, 1]
    x_min = 0.1
    x_max = 6.0
    t = (x - x_min) / (x_max - x_min)
    
    # Base angle
    theta = 2.0 * np.pi * t
    
    # Compute basis functions
    # Harmonic 1
    c1 = np.cos(theta)
    s1 = np.sin(theta)
    # Harmonic 2
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    # Harmonic 3
    c3 = np.cos(3.0 * theta)
    s3 = np.sin(3.0 * theta)
    # Harmonic 4
    c4 = np.cos(4.0 * theta)
    s4 = np.sin(4.0 * theta)
    # Harmonic 5 (Cosine only)
    c5 = np.cos(5.0 * theta)
    
    # Linear combination
    # Using dot product or explicit sum for clarity and speed
    # Explicit sum is fast enough for 200 points
    y = (params[0] + 
         params[1] * c1 + params[2] * s1 + 
         params[3] * c2 + params[4] * s2 + 
         params[5] * c3 + params[6] * s3 + 
         params[7] * c4 + params[8] * s4 + 
         params[9] * c5)
         
    return y
# EVOLVE-BLOCK-END
