"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a sum of Gaussian basis functions.
    
    Uses 10 parameters to define 3 Gaussian bumps in the (t, x, v) space.
    Structure:
    - 3 weights (w1, w2, w3)
    - 3 center scales (shared across dimensions for each bump)
    - 3 center offsets (one per dimension, shared across bumps)
    - 1 global offset
    
    Total: 3 + 3 + 3 + 1 = 10 parameters
    
    This allows capturing smooth nonlinear acceleration fields without 
    assuming specific polynomial or rational structures.
    """
    w1, w2, w3 = params[0], params[1], params[2]
    s1, s2, s3 = params[3], params[4], params[5]
    ct, cx, cv = params[6], params[7], params[8]
    offset = params[9]
    
    # Ensure scales are positive and reasonable to avoid numerical issues
    s1 = np.maximum(np.abs(s1), 0.1)
    s2 = np.maximum(np.abs(s2), 0.1)
    s3 = np.maximum(np.abs(s3), 0.1)
    
    # Compute Gaussian bumps
    # Each bump is centered at (ct, cx, cv) with scales s1, s2, s3
    # The Gaussian is: exp(-(t-ct)^2/(2*s1^2) - (x-cx)^2/(2*s2^2) - (v-cv)^2/(2*s3^2))
    
    dt = (t - ct) / s1
    dx = (x - cx) / s2
    dv = (v - cv) / s3
    
    exponent = -(dt**2 + dx**2 + dv**2) / 2.0
    gaussian = np.exp(exponent)
    
    # Sum of weighted Gaussians plus offset
    # Since we only have 3 weights but need to model the full field,
    # we use the same Gaussian shape but modulate by different functions
    # Actually, let's use a different structure: 3 Gaussians with shared centers
    # but different weights and slightly different scales
    
    # Better approach: Use 3 Gaussians with shared center but different scales
    # and weights, plus offset
    
    # Gaussian 1
    g1 = np.exp(-((t - ct)**2 / (2 * s1**2) + (x - cx)**2 / (2 * s1**2) + (v - cv)**2 / (2 * s1**2)))
    # Gaussian 2  
    g2 = np.exp(-((t - ct)**2 / (2 * s2**2) + (x - cx)**2 / (2 * s2**2) + (v - cv)**2 / (2 * s2**2)))
    # Gaussian 3
    g3 = np.exp(-((t - ct)**2 / (2 * s3**2) + (x - cx)**2 / (2 * s3**2) + (v - cv)**2 / (2 * s3**2)))
    
    return w1 * g1 + w2 * g2 + w3 * g3 + offset
# EVOLVE-BLOCK-END
